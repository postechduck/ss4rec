import torch
from torch import nn
from mamba_ssm import Mamba
from s5 import S5, S5Block
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import random
                    

class SS4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SS4Rec, self).__init__(config, dataset)

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

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
            
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
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, item_timeseq, pos_timestamps): 
        time_interval = torch.zeros_like(item_timeseq).to(item_seq.device) # [B, L]
        time_interval[:, :-1] = item_timeseq[:, 1:] - item_timeseq[:, :-1]
        
        indices = torch.arange(time_interval.size(0)).to(item_seq.device)
        last_item_indices = item_seq_len - 1
        time_interval[indices, last_item_indices] = pos_timestamps - item_timeseq[indices, last_item_indices]

        item_emb = self.item_embedding(item_seq) # [B, L, D] 

        # item_emb = self.dropout(item_emb)
        # item_emb = self.LayerNorm(item_emb)
        output = item_emb
        for i in range(self.num_layers):
            output, _ = self.model[i](output, time_interval)
        output = self.gather_indexes(output, item_seq_len - 1)

        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_timeseq = interaction[self.TIME_SEQ]
        pos_items = interaction[self.POS_ITEM_ID]
        pos_timestamps = interaction[self.POS_TIME]
        seq_output = self.forward(item_seq, item_seq_len, item_timeseq, pos_timestamps)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            # logits = torch.softmax(logits,dim=1)
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_timeseq = interaction[self.TIME_SEQ]
        test_item = interaction[self.ITEM_ID]
        timestamps = interaction[self.TIMESTAMP]
        
        # predict next n items
        self.next_n = 1
        indices = torch.arange(item_seq.size(0)).to(item_seq.device)
        for i in range(1, self.next_n):
            index = item_seq_len - i
            item_seq[indices, index] = 0
        
        for i in range(self.next_n): # 0 1
            seq_output = self.forward(item_seq, item_seq_len - (self.next_n - i + 1), item_timeseq, timestamps)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_timeseq = interaction[self.TIME_SEQ]
        timestamps = interaction[self.TIMESTAMP]
        
        # predict next n items
        # mask the last n-1 items in the sequence
        indices = torch.arange(item_seq.size(0)).to(item_seq.device)
        for i in range(1, self.next_n):
            index = item_seq_len - i
            item_seq[indices, index] = 0
     
        for i in range(self.next_n): # 0 1
            seq_output = self.forward(item_seq, item_seq_len - (self.next_n - i - 1), item_timeseq, timestamps)
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(
                seq_output, test_items_emb.transpose(0, 1)
            )  # [B, n_items]
            if i == 0:
                break
            pred_item = torch.argmax(scores, dim=1) # [B] predicted next item added to the sequence
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
                dt_min = dt_min,
                dt_max =dt_max,
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