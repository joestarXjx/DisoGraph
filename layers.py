import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, add_self_loops=False, normalize=True, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.add_self_loops = add_self_loops  # 是否添加自环
        self.normalize = normalize  # 是否进行邻接矩阵归一化

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # 偏置参数
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**(0.5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: node features with shape [B, L, D]
        adj: adjacency matrix with shape [B, L, L]
        """
        B, L, D = x.shape
        
        # 步骤1: 添加自环（A' = A + I）
        if self.add_self_loops:
            # 生成单位矩阵并扩展到批次维度 [B, L, L]
            self_loops = torch.eye(L, device=x.device).unsqueeze(0).repeat(B, 1, 1)
            adj = adj + self_loops  # 邻接矩阵加入自环

        # 步骤2: 邻接矩阵归一化（D^(-1/2) * A' * D^(-1/2)）
        if self.normalize:
            # 计算度矩阵 D（对角线元素为每行之和）
            degree = adj.sum(dim=2, keepdim=True)  # [B, L, 1]
            
            # 防止除零（度为0的节点设置为1）
            degree = torch.clamp(degree, min=1.0)
            
            # 计算 D^(-1/2)
            degree_inv_sqrt = degree **(-0.5)  # [B, L, 1]
            
            # 归一化邻接矩阵: D^(-1/2) * A' * D^(-1/2)
            adj = adj * degree_inv_sqrt  # [B, L, L] * [B, L, 1] → [B, L, L]
            adj = adj * degree_inv_sqrt.transpose(1, 2)  # [B, L, L] * [B, 1, L] → [B, L, L]

        
        # 步骤3: 图卷积核心计算（A' * X * W）
        # 特征变换: X * W → [B, L, D] * [D, out_features] → [B, L, out_features]
        support = torch.matmul(x, self.weight)
        # 邻接矩阵与特征相乘: A' * (X * W)
        out = torch.bmm(adj, support)  # [B, L, L] * [B, L, out_features] → [B, L, out_features]
        # 步骤4: 添加偏置
        if self.bias is not None:
            out = out + self.bias
        return out
    
''' Modifies an existing transformer and introduce the LoRA layers '''
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, scaling_rank, init_scale):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
            if init_scale < 0:
                self.lora_b = nn.Parameter(torch.randn(linear_layer.out_features, rank) * init_scale)
            else:
                self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank) * init_scale
                )
            else:
                self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features, self.scaling_rank))

    def forward(self, input):
        if self.scaling_rank == 1 and self.rank == 0:
            # parsimonious implementation for ia3 and lora scaling
            if self.multi_lora_a.requires_grad:
                hidden = F.linear((input * self.multi_lora_a.flatten()), self.weight, self.bias)
            else:
                hidden = F.linear(input, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten()
            return hidden
        else:
            # general implementation for lora (adding and scaling)
            weight = self.weight
            if self.scaling_rank:
                weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
            if self.rank:
                weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
            return F.linear(input, weight, self.bias)

class LoRAConfig:
    def __init__(self):
        self.lora_rank = 4
        self.lora_init_scale = 0.01
        self.lora_modules = ".*layer\.(2[7-9]|3[0-2])\.attention"
        self.lora_sub_modules = "self|output"
        self.lora_layers = "query|key|value|dense"
        self.lora_scaling_rank = 0
        self.trainable_param_names = ".*lora_[ab].*"

def modify_with_lora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_sub_modules, c_name):
                    for s_name, sub_layer in dict(layer.named_children()).items():
                        if re.fullmatch(config.lora_layers, s_name):
                            assert isinstance(
                                sub_layer, nn.Linear
                            ), f"LoRA can only be applied to torch.nn.Linear, but {sub_layer} is {type(sub_layer)}."
                            setattr(
                                layer,
                                s_name,
                                LoRALinear(sub_layer, config.lora_rank, config.lora_scaling_rank, config.lora_init_scale),
                            )
    return transformer