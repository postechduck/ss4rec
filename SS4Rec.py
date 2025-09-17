import torch
from torch import nn
from mamba_ssm import Mamba
from s5 import S5
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:
    pd = None


class SS4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SS4Rec, self).__init__(config, dataset)
        # ===== Basic fields =====
        self.next_n = config["next_n"]
        self.TIMESTAMP = config["TIMESTAMP_FIELD"]
        self.TIME_SEQ = self.TIMESTAMP + config["LIST_SUFFIX"]
        self.POS_TIME = self.TIMESTAMP

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # ===== SSM hyperparams =====
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.dt_min = config["dt_min"]
        self.dt_max = config["dt_max"]
        self.d_P = config["d_P"]
        self.d_H = config["d_H"]
        self.model_type = config["model_type"]

        # ===== Item embedding =====
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        # ===== Item genre multi-hot → Linear (additive) =====
        self.item_genre_enabled = True
        self.item_genre_vocab_size = None
        self.register_buffer("item_genre_mh", None)  # [n_items, G], row 0 (padding) is all zeros
        try:
            self._build_item_genre_table_from_dataset(dataset)
        except Exception as e:
            print(f"[WARN] item genre multi-hot disabled (reason: {e})")
            self.item_genre_enabled = False

        if self.item_genre_enabled and self.item_genre_vocab_size and self.item_genre_vocab_size > 0:
            # project G -> D; bias=False keeps it purely additive
            self.item_genre_linear = nn.Linear(self.item_genre_vocab_size, self.hidden_size, bias=False)
            self.item_genre_dropout = nn.Dropout(p=float(self.dropout_prob))
        else:
            self.item_genre_linear = nn.Identity()
            self.item_genre_dropout = nn.Identity()

        # ===== Blocks =====
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
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

        # ===== User metadata (multi-hot → Linear) =====
        self.user_hidden_size = int(config['user_hidden_size']) if 'user_hidden_size' in config else 32
        self.age_classes    = int(config['ml1m_age_classes'])    if 'ml1m_age_classes'    in config else 7
        self.gender_classes = int(config['ml1m_gender_classes']) if 'ml1m_gender_classes' in config else 2
        self.job_classes    = int(config['ml1m_job_classes'])    if 'ml1m_job_classes'    in config else 21
        total_user_multihot_dim = self.age_classes + self.gender_classes + self.job_classes
        self.user_feat_linear = nn.Linear(total_user_multihot_dim, self.user_hidden_size)
        self.user_dropout = nn.Dropout(p=float(self.dropout_prob))

        # ===== concat(user) → fuse(back to hidden_size) =====
        self.fuse = nn.Linear(self.hidden_size + self.user_hidden_size, self.hidden_size)
        self.fuse_ln = nn.LayerNorm(self.hidden_size)

        # ===== SSE hyperparams =====
        self.sse_user_p   = float(config['sse_user_p'])   if 'sse_user_p'   in config else 0.0
        self.sse_item_p   = float(config['sse_item_p'])   if 'sse_item_p'   in config else 0.0
        self.sse_pos_p    = float(config['sse_pos_p'])    if 'sse_pos_p'    in config else 0.0
        self.sse_item_mode = str(config['sse_item_mode']) if 'sse_item_mode' in config else 'batch'
        self.n_items_cached = getattr(self.item_embedding, "num_embeddings", None)

        self.apply(self._init_weights)

    # ===================== util: build genre table =====================
    def _build_item_genre_table_from_dataset(self, dataset):
        n_items = self.n_items
        item_id_field = self.ITEM_ID

        if not hasattr(dataset, "item_feat"):
            raise RuntimeError("dataset.item_feat not found")
        item_feat = dataset.item_feat

        # to pandas
        if hasattr(item_feat, "to_pandas"):
            df = item_feat.to_pandas()
        elif pd is not None:
            data_dict = {}
            for k in item_feat.interaction.keys():
                v = item_feat[k]
                try:
                    data_dict[k] = v.cpu().numpy()
                except Exception:
                    data_dict[k] = v
            df = pd.DataFrame(data_dict)
        else:
            raise RuntimeError("cannot convert item_feat to pandas DataFrame")

        # pick genre column
        genre_col_candidates = ["genre", "genres", "genre_seq", "genre_list"]
        genre_col = None
        for c in genre_col_candidates:
            if c in df.columns:
                genre_col = c
                break
        if genre_col is None:
            raise RuntimeError("no genre column found in item_feat")

        def split_genres(x):
            if x is None:
                return []
            if isinstance(x, (list, tuple)):
                return [str(t).strip() for t in x if str(t).strip() != ""]
            s = str(x)
            for sep in ["|", ",", ";", "/"]:
                if sep in s:
                    return [p.strip() for p in s.split(sep) if p.strip()]
            return [s.strip()] if s.strip() else []

        # build vocab
        vocab = {}
        for val in df[genre_col].tolist():
            for g in split_genres(val):
                if g not in vocab:
                    vocab[g] = len(vocab)
        G = len(vocab)
        if G == 0:
            raise RuntimeError("empty genre vocabulary")

        # multi-hot table
        mh = torch.zeros((n_items, G), dtype=torch.float32)  # row 0 stays zeros (padding)
        # fix item id column if needed
        if item_id_field not in df.columns:
            for cand in [self.ITEM_ID_FIELD, "item_id", "item", "sid", "iid"]:
                if cand in df.columns:
                    item_id_field = cand
                    break

        for _, row in df.iterrows():
            try:
                iid = int(row[item_id_field])
            except Exception:
                continue
            if iid <= 0 or iid >= n_items:
                continue
            genres = split_genres(row[genre_col])
            for g in genres:
                if g in vocab:
                    mh[iid, vocab[g]] = 1.0

        self.item_genre_vocab_size = G
        self.register_buffer("item_genre_mh", mh)

    # ===================== weights init =====================
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # ===================== user feature helpers =====================
    def _pick_user_column(self, interaction, names):
        for n in names:
            if n in interaction:
                return interaction[n]
        return None

    def _build_user_emb(self, age_idx, gender_idx, job_idx):
        age_1h    = F.one_hot(age_idx,    num_classes=self.age_classes).float()     # [B, A]
        gender_1h = F.one_hot(gender_idx, num_classes=self.gender_classes).float()  # [B, G]
        job_1h    = F.one_hot(job_idx,    num_classes=self.job_classes).float()     # [B, J]
        user_multihot = torch.cat([age_1h, gender_1h, job_1h], dim=-1)              # [B, A+G+J]
        user_emb = self.user_feat_linear(user_multihot)                              # [B, d_user]
        user_emb = self.user_dropout(user_emb)
        return user_emb

    # ===================== item representation for targets =====================
    def _all_item_repr(self):
        """Return [n_items, D] table for candidates (id + genre). Deterministic (no dropout)."""
        W = self.item_embedding.weight  # [n_items, D]
        if self.item_genre_enabled and self.item_genre_mh is not None:
            Gproj = self.item_genre_linear(self.item_genre_mh)  # [n_items, D]
            return W + Gproj
        return W

    def _item_repr(self, item_ids):
        """Return representation for specific item ids (id + genre)."""
        emb = self.item_embedding(item_ids)  # [..., D]
        if self.item_genre_enabled and self.item_genre_mh is not None:
            g = self.item_genre_linear(self.item_genre_mh[item_ids])  # [..., D]
            return emb + g
        return emb

    # ===================== forward =====================
    def forward(self, item_seq, item_seq_len, item_timeseq, pos_timestamps,
                age_idx=None, gender_idx=None, job_idx=None):
        device = item_seq.device
        B, L = item_seq.size(0), item_seq.size(1)

        # time intervals
        time_interval = torch.zeros_like(item_timeseq, device=device)  # [B, L]
        time_interval[:, :-1] = item_timeseq[:, 1:] - item_timeseq[:, :-1]
        idx = torch.arange(B, device=device)
        last = item_seq_len - 1
        time_interval[idx, last] = pos_timestamps - item_timeseq[idx, last]

        # valid mask (padding==0 → False)
        valid_mask = (item_seq != 0)  # [B, L], bool

        # item id embedding
        item_id_emb = self.item_embedding(item_seq)  # [B, L, D]

        # item genre projection (additive), padding row => zeros
        if self.item_genre_enabled and self.item_genre_mh is not None:
            genre_mh = self.item_genre_mh[item_seq]                 # [B, L, G]
            genre_proj = self.item_genre_linear(genre_mh)           # [B, L, D]
            genre_proj = self.item_genre_dropout(genre_proj)
            seq_emb = item_id_emb + genre_proj                      # [B, L, D]
        else:
            seq_emb = item_id_emb

        # user emb (batch-level)
        if age_idx is None:    age_idx    = torch.zeros(B, dtype=torch.long, device=device)
        if gender_idx is None: gender_idx = torch.zeros(B, dtype=torch.long, device=device)
        if job_idx is None:    job_idx    = torch.zeros(B, dtype=torch.long, device=device)

        user_emb = self._build_user_emb(age_idx, gender_idx, job_idx)  # [B, d_user]

        # SSE (user side): batch permutation
        if self.training and self.sse_user_p > 0.0:
            mask_u = (torch.rand(B, device=device) < self.sse_user_p)  # [B]
            if mask_u.any():
                perm_b = torch.randperm(B, device=device)
                user_emb_perm = user_emb[perm_b]
                user_emb = torch.where(mask_u.unsqueeze(-1), user_emb_perm, user_emb)

        # concat user → fuse back to hidden_size
        user_seq = user_emb.unsqueeze(1).expand(B, L, self.user_hidden_size)  # [B, L, d_user]
        fused = torch.cat([seq_emb, user_seq], dim=-1)                        # [B, L, D+d_user]
        seq_emb = self.fuse(fused)                                            # [B, L, D]
        seq_emb = self.fuse_ln(seq_emb)

        # SSE (item side) with padding-safe masking
        if self.training and self.sse_item_p > 0.0:
            rand_mask = (torch.rand(B, L, device=device) < self.sse_item_p)   # [B, L]
            sse_mask  = rand_mask & valid_mask                                # padding excluded
            if sse_mask.any():
                if self.sse_item_mode == "batch":
                    # shuffle across batch for same time step
                    perm_b = torch.randperm(B, device=device)
                    seq_emb_perm = seq_emb[perm_b]                            # [B, L, D]
                    seq_emb = torch.where(sse_mask.unsqueeze(-1), seq_emb_perm, seq_emb)
                else:
                    # global random item embeddings (avoid padding=0)
                    n_items = self.n_items_cached or self.item_embedding.num_embeddings
                    rand_ids = torch.randint(1, n_items, (B, L), device=device)
                    rand_emb = self.item_embedding(rand_ids)                  # [B, L, D]
                    seq_emb = torch.where(sse_mask.unsqueeze(-1), rand_emb, seq_emb)

        # force padding steps to zero (extra safety)
        seq_emb = seq_emb * valid_mask.unsqueeze(-1)

        # backbone
        output = seq_emb
        for i in range(self.num_layers):
            output, _ = self.model[i](output, time_interval)
        output = self.gather_indexes(output, item_seq_len - 1)  # [B, D]
        return output

    # ===================== loss =====================
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_timeseq = interaction[self.TIME_SEQ]
        pos_items = interaction[self.POS_ITEM_ID]
        pos_timestamps = interaction[self.POS_TIME]

        # user features
        age_idx    = self._pick_user_column(interaction, ['age_idx', 'age'])
        gender_idx = self._pick_user_column(interaction, ['gender_idx', 'gender'])
        job_idx    = self._pick_user_column(interaction, ['job_idx', 'occupation', 'job'])

        seq_output = self.forward(
            item_seq, item_seq_len, item_timeseq, pos_timestamps,
            age_idx=age_idx, gender_idx=gender_idx, job_idx=job_idx
        )

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]

            # use id + genre for targets
            pos_items_emb = self._item_repr(pos_items)  # [B, D]
            neg_items_emb = self._item_repr(neg_items)  # [B, D]

            # optional SSE on positive target
            if self.training and self.sse_pos_p > 0.0:
                B = pos_items_emb.size(0)
                device = pos_items_emb.device
                mask_b = (torch.rand(B, device=device) < self.sse_pos_p)
                if mask_b.any():
                    n_items = self.item_embedding.num_embeddings
                    alt_ids = torch.randint(low=1, high=n_items, size=(int(mask_b.sum()),), device=device)
                    pos_ids_masked = pos_items[mask_b]
                    alt_ids = torch.where(alt_ids == pos_ids_masked, (alt_ids + 1) % n_items, alt_ids)
                    pos_items_emb[mask_b] = self._item_repr(alt_ids)

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss

        else:  # CrossEntropy
            # candidate table includes id + genre; deterministic (no dropout)
            W_all = self._all_item_repr()  # [n_items, D]
            o = seq_output                 # [B, D]

            if self.training and self.sse_pos_p > 0.0:
                B = o.size(0)
                device = o.device
                mask_b = (torch.rand(B, device=device) < self.sse_pos_p)
                if mask_b.any():
                    n_items = W_all.size(0)
                    pos_ids_masked = pos_items[mask_b]
                    alt_ids = torch.randint(low=1, high=n_items, size=(int(mask_b.sum()),), device=device)
                    alt_ids = torch.where(alt_ids == pos_ids_masked, (alt_ids + 1) % n_items, alt_ids)
                    W_mod = W_all.clone()
                    W_mod[pos_ids_masked] = W_all[alt_ids]
                    logits = torch.matmul(o, W_mod.transpose(0, 1))
                else:
                    logits = torch.matmul(o, W_all.transpose(0, 1))
            else:
                logits = torch.matmul(o, W_all.transpose(0, 1))

            loss = self.loss_fct(logits, pos_items)
            return loss

    # ===================== predict =====================
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_timeseq = interaction[self.TIME_SEQ]
        test_item = interaction[self.ITEM_ID]
        timestamps = interaction[self.TIMESTAMP]

        age_idx    = self._pick_user_column(interaction, ['age_idx', 'age'])
        gender_idx = self._pick_user_column(interaction, ['gender_idx', 'gender'])
        job_idx    = self._pick_user_column(interaction, ['job_idx', 'occupation', 'job'])

        # predict next_n
        self.next_n = 1
        indices = torch.arange(item_seq.size(0)).to(item_seq.device)
        for i in range(1, self.next_n):
            index = item_seq_len - i
            item_seq[indices, index] = 0

        for i in range(self.next_n):
            seq_output = self.forward(
                item_seq, item_seq_len - (self.next_n - i + 1), item_timeseq, timestamps,
                age_idx=age_idx, gender_idx=gender_idx, job_idx=job_idx
            )
        test_item_emb = self._item_repr(test_item)  # id + genre
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    # ===================== full-sort predict =====================
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_timeseq = interaction[self.TIME_SEQ]
        timestamps = interaction[self.TIMESTAMP]

        age_idx    = self._pick_user_column(interaction, ['age_idx', 'age'])
        gender_idx = self._pick_user_column(interaction, ['gender_idx', 'gender'])
        job_idx    = self._pick_user_column(interaction, ['job_idx', 'occupation', 'job'])

        indices = torch.arange(item_seq.size(0)).to(item_seq.device)
        for i in range(1, self.next_n):
            index = item_seq_len - i
            item_seq[indices, index] = 0

        for i in range(self.next_n):
            seq_output = self.forward(
                item_seq, item_seq_len - (self.next_n - i - 1), item_timeseq, timestamps,
                age_idx=age_idx, gender_idx=gender_idx, job_idx=job_idx
            )
            test_items_emb = self._all_item_repr()  # [n_items, D], id + genre
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
            if i == 0:
                break
            pred_item = torch.argmax(scores, dim=1)
            index = item_seq_len - (self.next_n - i - 1)
            item_seq[indices, index] = pred_item[indices]
        return scores


class SSBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, dt_min, dt_max, d_P, d_H, expand, dropout, **kwargs):
        super().__init__()
        self.model_type = kwargs["model_type"]
        self.S5 = S5(width=d_H, state_width=d_P, dt_min=dt_min, dt_max=dt_max)
        self.S6 = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor, time_interval):
        S5_output = self.S5(input_tensor, time_interval)
        S5_output = self.dropout(S5_output)
        S5_output = self.LayerNorm(S5_output + input_tensor)

        output_states = self.S6(S5_output)
        output_states = self.dropout(output_states)
        output_states = self.LayerNorm(output_states + S5_output)
        return output_states, time_interval
