import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import EsmModel
from transformers.modeling_outputs import TokenClassifierOutput
from layers import GraphConvolution

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EsmModel(config, add_pooling_layer=False)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(config.hidden_size, 512)
        self.fc2 = nn.Linear(512,2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask, labels = None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs[0]  # B,L,D

        logits = self.fc2(self.dropout(self.relu(self.fc1(x))))
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits =  logits.view(-1, 2)
            active_labels  = torch.where(active_loss, labels.view(-1), torch.tensor(-100).type_as(labels))

            valid_logits = active_logits[active_labels != -100]
            valid_labels = active_labels[active_labels != -100]

            loss = loss_fct(valid_logits, valid_labels)
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

class DisoGraph(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EsmModel(config, add_pooling_layer=False)
        
        self.gcn1 = GraphConvolution(config.hidden_size, 512)
        self.gcn2 = GraphConvolution(512,512)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512,2)
        self.dropout = nn.Dropout(0.2)

    # 序列邻接性邻接矩阵
    def sequence_proximity_adj(self, adj, input_ids, window_size):
        B, _ = input_ids.shape
        batch_lens = (input_ids != 1).sum(1)
        for b in range(B):
            len = batch_lens[b]
            for i in range(len):
                start = max(0, i-window_size)
                end = min(len, i+window_size+1)
                adj[b,i,start:end] = 1
        return adj

    # 分段处理序列并预测接触
    def predict_contacts_from_batch(self, input_ids, attention_mask, chunk_size):
        B, L = input_ids.shape
        batch_lens = (input_ids != 1).sum(1)

        if L <= chunk_size:
            attn_weight = self.encoder(input_ids, attention_mask,return_dict=True, output_attentions=True).attentions
            attentions = torch.stack(attn_weight,dim=1)
            contact_maps = self.encoder.contact_head(input_ids,attentions)
            contact_maps = F.pad(contact_maps, pad=(1, 1, 1, 1), value=0)
            return contact_maps

        contact_maps = torch.zeros((B, L, L), dtype=torch.float32, device=input_ids.device)
        count_maps = torch.zeros((B, L, L), dtype=torch.float32, device=input_ids.device)
        for b in range(B):
            tokens = input_ids[b,1:batch_lens[b]-1]  # seq_len
            overlap = chunk_size // 2
            valid_len = batch_lens[b] - 2
            start_positions = list(range(0, valid_len, chunk_size - overlap))
            for start in start_positions:
                end = min(start + chunk_size, valid_len)
                # 提取当前窗口的token和attention, 窗口token前面添加<cls>,最后添加<eos>
                window_tokens = torch.cat((torch.tensor([0]).to(tokens.device), tokens[start:end], torch.tensor([2]).to(tokens.device)),dim=-1)
                window_tokens = torch.unsqueeze(window_tokens, dim=0) # 1, chunk_size+2
                window_attentions_weights = self.encoder(window_tokens,return_dict=True, output_attentions=True).attentions
                window_attentions = torch.stack(window_attentions_weights, dim=1)
                # 预测当前窗口
                window_contact = self.encoder.contact_head(window_tokens,window_attentions)  #1 , chunk_size, chunk_size
                # 将窗口接触图添加到总图中
                contact_maps[b, start:end, start:end] += window_contact[0]
                count_maps[b, start:end, start:end] += 1
        
        contact_maps = torch.where(count_maps > 0, contact_maps / count_maps, contact_maps)
        threshold = 0.5
        contact_maps = (contact_maps >= threshold).int()
        return contact_maps 
    
    def forward(self, input_ids, attention_mask, labels = None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        residue_representations = outputs[0]  # B,L,D

        # Graph
        adj = self.predict_contacts_from_batch(input_ids,attention_mask, 128)
        adj = self.sequence_proximity_adj(adj, input_ids, 2)
        x = residue_representations
        x1 = self.gcn1(x,adj)
        x = F.relu(x1)
        x2 = self.gcn2(x,adj)
        x = torch.cat([x1,x2],dim=-1)
        
        logits = self.fc2(self.dropout(self.relu(self.fc1(x))))
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits =  logits.view(-1, 2)
            active_labels  = torch.where(active_loss, labels.view(-1), torch.tensor(-100).type_as(labels))

            valid_logits = active_logits[active_labels != -100]
            valid_labels = active_labels[active_labels != -100]

            loss = loss_fct(valid_logits, valid_labels)
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )