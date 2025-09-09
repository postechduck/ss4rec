import torch
from torch import nn
from mamba_ssm import Mamba
from s5 import S5, S5Block
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn.functional as F
import random

class SS4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SS4Rec, self).__init__(config, dataset)
        self.next_n = config["next_n"]  # 추가 코드
        self.TIMESTAMP = config["TIMESTAMP_FIELD"]
        self.TIME_SEQ = self.TIMESTAMP + config["LIST_SUFFIX"]
        self.POS_TIME = self.TIMESTAMP
        
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        
        # Hyperparameters for SSM
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.dt_min = config["dt_min"]
        self.dt_max = config["dt_max"]
        self.d_P = config["d_P"]
        self.d_H = config["d_H"]
        self.model_type = config["model_type"]

        # ===== Embeddings =====
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
            
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        # ===== SSM blocks =====
        self.model = nn.ModuleList([
            SSBlock(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                d_P=self.d_P,
                d_H=self.d_H,
                expand=self.expand,
                dropout=self.dropout_prob,
                model_type=self.model_type,
            ) for _ in range(self.num_layers)
        ])
        
        # ===== Loss =====
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # === ADDED: user multi-hot → Linear(learnable) ===
        self.user_hidden_size = int(config['user_hidden_size']) if 'user_hidden_size' in config else 32
        self.age_classes    = int(config['ml1m_age_classes'])    if 'ml1m_age_classes'    in config else 7
        self.gender_classes = int(config['ml1m_gender_classes']) if 'ml1m_gender_classes' in config else 2
        self.job_classes    = int(config['ml1m_job_classes'])    if 'ml1m_job_classes'    in config else 21

        total_user_multihot_dim = self.age_classes + self.gender_classes + self.job_classes
        self.user_feat_linear = nn.Linear(total_user_multihot_dim, self.user_hidden_size)
        self.user_dropout = nn.Dropout(p=float(self.dropout_prob))

        # === ADDED: concat + fuse back to hidden_size ===
        self.fuse = nn.Linear(self.hidden_size + self.user_hidden_size, self.hidden_size)
        self.fuse_ln = nn.LayerNorm(self.hidden_size)

        # === ADDED: SSE hyperparams ===
        self.sse_user_p   = float(config['sse_user_p'])   if 'sse_user_p'   in config else 0.0
        self.sse_item_p   = float(config['sse_item_p'])   if 'sse_item_p'   in config else 0.0
        self.sse_pos_p = float(config['sse_pos_p']) if 'sse_pos_p'   in config else 0.0
        self.sse_item_mode = str(config['sse_item_mode']) if 'sse_item_mode' in config else 'global'  # or 'batch'
        # (cache item vocab size for global item SSE)
        self.n_items_cached = getattr(self.item_embedding, "num_embeddings", None)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # === ADDED: small helper to pick user columns flexibly ===
    def _pick_user_column(self, interaction, names):
        for n in names:
            if n in interaction:
                return interaction[n]
        return None

    # === ADDED: build user embedding from multi-hot ===
    def _build_user_emb(self, age_idx, gender_idx, job_idx):
        # one-hot & concat
        age_1h    = F.one_hot(age_idx,    num_classes=self.age_classes).float()       # [B, A]
        gender_1h = F.one_hot(gender_idx, num_classes=self.gender_classes).float()    # [B, G]
        job_1h    = F.one_hot(job_idx,    num_classes=self.job_classes).float()       # [B, J]
        user_multihot = torch.cat([age_1h, gender_1h, job_1h], dim=-1)                # [B, A+G+J]
        # linear proj (+ learnable)
        user_emb = self.user_feat_linear(user_multihot)                                # [B, d_user]
        user_emb = self.user_dropout(user_emb)
        return user_emb

    # === CHANGED: forward에 user feature를 받도록 확장 ===
    def forward(self, item_seq, item_seq_len, item_timeseq, pos_timestamps,
                age_idx=None, gender_idx=None, job_idx=None): 
        device = item_seq.device
        B = item_seq.size(0)

        # time interval
        time_interval = torch.zeros_like(item_timeseq).to(device) # [B, L]
        time_interval[:, :-1] = item_timeseq[:, 1:] - item_timeseq[:, :-1]
        indices = torch.arange(time_interval.size(0), device=device)
        last_item_indices = item_seq_len - 1
        time_interval[indices, last_item_indices] = pos_timestamps - item_timeseq[indices, last_item_indices]

        # item embeddings
        seq_emb = self.item_embedding(item_seq) # [B, L, D]

        # ===== ADDED: user embedding from (age, gender, job) =====
        # fallback to zeros if not provided
        if age_idx is None:    age_idx    = torch.zeros(B, dtype=torch.long, device=device)
        if gender_idx is None: gender_idx = torch.zeros(B, dtype=torch.long, device=device)
        if job_idx is None:    job_idx    = torch.zeros(B, dtype=torch.long, device=device)

        user_emb = self._build_user_emb(age_idx, gender_idx, job_idx)   # [B, d_user]

        # ===== ADDED: SSE - user side (torch.where → gradient 유지) =====
        if self.training and self.sse_user_p > 0.0:
            mask_u = (torch.rand(B, device=device) < self.sse_user_p)   # [B] bool
            if mask_u.any():
                perm_b = torch.randperm(B, device=device)
                user_emb_perm = user_emb[perm_b]                        # [B, d_user]
                user_emb = torch.where(mask_u.unsqueeze(-1), user_emb_perm, user_emb)

        # ===== ADDED: concat user → fuse back to hidden_size =====
        L = seq_emb.size(1)
        user_seq = user_emb.unsqueeze(1).expand(B, L, self.user_hidden_size)  # [B, L, d_user]
        fused = torch.cat([seq_emb, user_seq], dim=-1)                        # [B, L, D+d_user]
        seq_emb = self.fuse(fused)                                            # [B, L, D]
        seq_emb = self.fuse_ln(seq_emb)

        # ===== ADDED: SSE - item side =====
        if self.training and self.sse_item_p > 0.0:
            mask_x = (torch.rand(B, L, device=device) < self.sse_item_p)      # [B, L] bool
            if mask_x.any():
                mode = getattr(self, "sse_item_mode", "batch")
                if mode == "batch":
                    # 배치 차원만 셔플 (동일 time step 전체를 다른 유저의 것으로)
                    perm_b = torch.randperm(B, device=device)                 # [B]
                    seq_emb_perm = seq_emb[perm_b]                            # [B, L, D]
                    seq_emb = torch.where(mask_x.unsqueeze(-1), seq_emb_perm, seq_emb)
                else:
                    # 'global' (or fallback): 임의 아이템 임베딩으로 대체
                    n_items = self.n_items_cached or self.item_embedding.num_embeddings
                    rand_ids = torch.randint(0, n_items, (B, L), device=device)
                    rand_emb = self.item_embedding(rand_ids)                  # [B, L, D]
                    seq_emb = torch.where(mask_x.unsqueeze(-1), rand_emb, seq_emb)

        # ===== original SSM backbone =====
        output = seq_emb
        for i in range(self.num_layers):
            output, _ = self.model[i](output, time_interval)
        output = self.gather_indexes(output, item_seq_len - 1)  # [B, D]

        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_timeseq = interaction[self.TIME_SEQ]
        pos_items = interaction[self.POS_ITEM_ID]
        pos_timestamps = interaction[self.POS_TIME]

        # === ADDED: grab user features from interaction (name-agnostic) ===
        age_idx    = self._pick_user_column(interaction, ['age_idx', 'age'])
        gender_idx = self._pick_user_column(interaction, ['gender_idx', 'gender'])
        job_idx    = self._pick_user_column(interaction, ['job_idx', 'occupation', 'job'])

        seq_output = self.forward(item_seq, item_seq_len, item_timeseq, pos_timestamps,
                                  age_idx=age_idx, gender_idx=gender_idx, job_idx=job_idx)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            # === SSE on POS-ITEM (BPR) : START ===
            if self.training and getattr(self, "sse_pos_p", 0.0) > 0.0:
                B = pos_items_emb.size(0)
                device = pos_items_emb.device
                mask_b = (torch.rand(B, device=device) < self.sse_pos_p)  # [B] bool
                if mask_b.any():
                    n_items = self.item_embedding.num_embeddings
                    # padding=0 회피: [1, n_items)에서 샘플
                    alt_ids = torch.randint(low=1, high=n_items, size=(int(mask_b.sum()),), device=device)
                    # (안전) 우연히 동일 id가 뽑히면 다음 id로 치환
                    pos_ids_masked = pos_items[mask_b]
                    alt_ids = torch.where(alt_ids == pos_ids_masked, (alt_ids + 1) % n_items, alt_ids)
                    alt_emb = self.item_embedding(alt_ids)                 # [M, D]
                    # 선택적으로 pos 임베딩 치환
                    pos_items_emb[mask_b] = alt_emb
            # === SSE on POS-ITEM (BPR) : END ===
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            W = self.item_embedding.weight  # [n_items, D]
            o = seq_output                  # [B, D]

            # === SSE on POS-ITEM (CE) : START ===
            if self.training and getattr(self, "sse_pos_p", 0.0) > 0.0:
                B = o.size(0)
                device = o.device
                mask_b = (torch.rand(B, device=device) < self.sse_pos_p)  # [B] bool
                if mask_b.any():
                    n_items = self.item_embedding.num_embeddings
                    pos_ids_masked = pos_items[mask_b]                    # [M]
                    alt_ids = torch.randint(low=1, high=n_items, size=(int(mask_b.sum()),), device=device)
                    alt_ids = torch.where(alt_ids == pos_ids_masked, (alt_ids + 1) % n_items, alt_ids)

                    # 주의: W를 직접 in-place 수정하지 않고, 사본 위에서만 치환
                    W_mod = W.clone()
                    W_mod[pos_ids_masked] = W[alt_ids]                   # 정답 행(row) 일시 치환

                    logits = torch.matmul(o, W_mod.transpose(0, 1))      # [B, n_items]
                else:
                    logits = torch.matmul(o, W.transpose(0, 1))
            else:
                logits = torch.matmul(o, W.transpose(0, 1))
            # === SSE on POS-ITEM (CE) : END ===

            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_timeseq = interaction[self.TIME_SEQ]
        test_item = interaction[self.ITEM_ID]
        timestamps = interaction[self.TIMESTAMP]

        # === ADDED: user features ===
        age_idx    = self._pick_user_column(interaction, ['age_idx', 'age'])
        gender_idx = self._pick_user_column(interaction, ['gender_idx', 'gender'])
        job_idx    = self._pick_user_column(interaction, ['job_idx', 'occupation', 'job'])
        
        # predict next n items
        self.next_n = 1
        indices = torch.arange(item_seq.size(0)).to(item_seq.device)
        for i in range(1, self.next_n):
            index = item_seq_len - i
            item_seq[indices, index] = 0
        
        for i in range(self.next_n):  # 0 1
            seq_output = self.forward(
                item_seq, item_seq_len - (self.next_n - i + 1), item_timeseq, timestamps,
                age_idx=age_idx, gender_idx=gender_idx, job_idx=job_idx
            )
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_timeseq = interaction[self.TIME_SEQ]
        timestamps = interaction[self.TIMESTAMP]

        # === ADDED: user features ===
        age_idx    = self._pick_user_column(interaction, ['age_idx', 'age'])
        gender_idx = self._pick_user_column(interaction, ['gender_idx', 'gender'])
        job_idx    = self._pick_user_column(interaction, ['job_idx', 'occupation', 'job'])
        
        # predict next n items
        # mask the last n-1 items in the sequence
        indices = torch.arange(item_seq.size(0)).to(item_seq.device)
        for i in range(1, self.next_n):
            index = item_seq_len - i
            item_seq[indices, index] = 0
     
        for i in range(self.next_n):  # 0 1
            seq_output = self.forward(
                item_seq, item_seq_len - (self.next_n - i - 1), item_timeseq, timestamps,
                age_idx=age_idx, gender_idx=gender_idx, job_idx=job_idx
            )
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(
                seq_output, test_items_emb.transpose(0, 1)
            )  # [B, n_items]
            if i == 0:
                break
            pred_item = torch.argmax(scores, dim=1)  # [B] predicted next item added to the sequence
            index = item_seq_len - (self.next_n - i - 1)
            item_seq[indices, index] = pred_item[indices]  
        
        return scores
    

class SSBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, dt_min, dt_max, d_P, d_H, expand, dropout, **kwargs):
        super().__init__()
        self.model_type = kwargs["model_type"]        
        
        self.S5 = S5(
            width=d_H,
            state_width=d_P,
            dt_min=dt_min,
            dt_max=dt_max,
        )

        self.S6 = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
    
    def forward(self, input_tensor, time_interval):
        S5_output = self.S5(input_tensor, time_interval)
        # add & norm
        S5_output = self.dropout(S5_output)
        S5_output = self.LayerNorm(S5_output + input_tensor)
        
        output_states = self.S6(S5_output)
        # add & norm
        output_states = self.dropout(output_states)
        output_states = self.LayerNorm(output_states + S5_output)
        
        return output_states, time_interval
