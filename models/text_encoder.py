import sys
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Tokenizer 

from lib.DeBERTaLib.deberta import DeBERTa
from lib.DeBERTaLib.config import ModelConfig

# =============================================================================
# 1. 辅助模块：注意力汇聚 (Attention Pooling)
# =============================================================================
class TextAttentionPooler(nn.Module):
    """
    将文本序列 [Batch, Len, Dim] 聚合成全局向量 [Batch, Dim]。
    修复了 FP16 下 Softmax 溢出的问题。
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, mask):
        # 1. 预处理：防止输入 x 已经包含 NaN (后期训练常见)
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        # 2. 计算原始分数
        scores = self.attn(x).squeeze(-1) # [B, L]
        
        # 3. Mask padding
        if mask is not None:
            # 确保 mask 也是 float 类型以匹配 scores
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # 4. 数值稳定性处理 (防止 Softmax 溢出)
        # 使用 keepdim=True 方便广播
        scores_max = scores.max(dim=1, keepdim=True)[0].detach()
        # 增加一个 epsilon 防止 scores_max 本身是极小值
        scores = scores - scores_max
        
        # 5. 限制范围，防止指数爆炸
        scores = torch.clamp(scores, min=-50, max=50)

        # 6. 计算权重
        attn_weights = F.softmax(scores, dim=-1) # [B, L]
        
        # --- [新增] 安全检查：如果某一行全为 NaN (虽然 clamp 了但以防万一) ---
        if torch.isnan(attn_weights).any():
            # 如果权重失效，退化为平均池化，不至于让梯度断掉或爆炸
            attn_weights = torch.ones_like(attn_weights) / attn_weights.size(-1)

        attn_weights = attn_weights.unsqueeze(-1) # [B, L, 1]
        
        # 7. 加权求和
        pooled = torch.sum(x * attn_weights, dim=1) # [B, D]
        
        return pooled, attn_weights
    


class StructuredPositionPointerEncoder(nn.Module):
    """Learn structured spatial relation pointers and relation graph embeddings from spatial tokens."""
    def __init__(self, hidden_dim, num_pointers=4, num_relation_templates=16):
        super().__init__()
        self.num_pointers = num_pointers
        self.num_relation_templates = num_relation_templates
        self.pointer_scorer = nn.Linear(hidden_dim, num_pointers)
        self.pointer_norm = nn.LayerNorm(hidden_dim)

        self.graph_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.graph_norm = nn.LayerNorm(hidden_dim)

        # relative_position = [dx, dy, orientation]
        self.rel_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh()
        )

        self.rel_token_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.out_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # Spatial Relation Graph Embedding: template bank for relation words/patterns.
        self.template_bank = nn.Parameter(torch.randn(num_relation_templates, hidden_dim) * 0.02)
        self.template_query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.template_logits_head = nn.Linear(hidden_dim, num_relation_templates)
        self.template_norm = nn.LayerNorm(hidden_dim)

    def forward(self, spa_feat, mask):
        # spa_feat: [B, L, D], mask: [B, L]
        pointer_logits = self.pointer_scorer(spa_feat).transpose(1, 2)  # [B, K, L]

        if mask is not None:
            m = mask.unsqueeze(1) == 0
            pointer_logits = pointer_logits.masked_fill(m, -1e4)

        pointer_attn = torch.softmax(pointer_logits, dim=-1)
        pointer_tokens = torch.matmul(pointer_attn, spa_feat)  # [B, K, D]
        pointer_tokens = self.pointer_norm(pointer_tokens)

        graph_tokens, _ = self.graph_attn(pointer_tokens, pointer_tokens, pointer_tokens)
        graph_tokens = self.graph_norm(pointer_tokens + graph_tokens)

        # Build pairwise relation features between neighboring pointers
        src = graph_tokens[:, :-1, :]
        dst = graph_tokens[:, 1:, :]
        pair_feat = torch.cat([src, dst], dim=-1)
        relative_position = self.rel_head(pair_feat)  # [B, K-1, 3]

        rel_tokens = self.rel_token_proj(torch.cat([src, relative_position], dim=-1))

        relation_seed = rel_tokens.mean(dim=1)
        template_query = self.template_query(relation_seed)
        template_logits = self.template_logits_head(template_query)
        template_attn = torch.softmax(template_logits, dim=-1)
        template_vec = torch.matmul(template_attn, self.template_bank)
        template_vec = self.template_norm(template_vec)

        relation_vec = self.out_fuse(torch.cat([graph_tokens.mean(dim=1), rel_tokens.mean(dim=1)], dim=-1))
        relation_vec = self.template_norm(relation_vec + template_vec)

        return {
            'pointer_tokens': pointer_tokens,
            'graph_tokens': graph_tokens,
            'relative_position': relative_position,
            'relation_tokens': rel_tokens,
            'relation_vec': relation_vec,
            'template_logits': template_logits,
            'template_attn': template_attn,
            'template_vec': template_vec,
        }
    
class BranchAdapter(nn.Module):
    """轻量分支适配器：以较小参数开销补充分支特化能力。"""
    def __init__(self, hidden_size, reduction=8, p=0.1):
        super().__init__()
        mid = max(hidden_size // reduction, 64)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(mid, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.norm(x + self.net(x))

# =============================================================================
# 2. 主模型：三分支 DeBERTa
# =============================================================================
class TriStreamDeBERTa(nn.Module):
    def __init__(self, model_path='./pretrain_ckp/deberta-v3-large', max_len=77):
        super().__init__()
        self.max_len = max_len
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        
        # 1. Load Config
        config_path = os.path.join(model_path, 'config.json')
        self.config = ModelConfig.from_json_file(config_path)
        
        # 2. Instantiate Raw Model
        raw_model = DeBERTa(config=self.config)
        
        # 3. Load Weights (Standard DeBERTa weights)
        self._load_weights(model_path, raw_model)
        
        # 4. Split Layers
        self.total_layers = len(raw_model.encoder.layer)
        self.split_idx = self.total_layers // 2 
        
        self.embeddings = raw_model.embeddings
        
        # A. Shared Encoder (底层通用特征)
        self.shared_encoder = nn.ModuleList([
            raw_model.encoder.layer[i] for i in range(self.split_idx)
        ])
        
        # B. Semantic Branch (语义/类别分支)
        self.semantic_encoder = nn.ModuleList([
            raw_model.encoder.layer[i] for i in range(self.split_idx, self.total_layers)
        ])
        
        # C. Spatial Branch (空间/位置分支) - DeepCopy 初始化
        self.spatial_encoder = copy.deepcopy(self.semantic_encoder)
        
        # D. Attribute Branch (属性/特征分支) - DeepCopy 初始化
        self.attribute_encoder = copy.deepcopy(self.semantic_encoder)
        
        # 共享组件
        self.rel_embeddings = raw_model.encoder.rel_embeddings
        self.rel_layer_norm = raw_model.encoder.LayerNorm 

        # Clean up
        del raw_model

        # 5. Task-Specific Heads (用于 Loss 约束)
        self.hidden_dim = self.config.hidden_size

        self.sem_adapter = BranchAdapter(self.hidden_dim)
        self.spa_adapter = BranchAdapter(self.hidden_dim)
        self.attr_adapter = BranchAdapter(self.hidden_dim)

        # 汇聚层 (每个分支一个)
        self.sem_pooler = TextAttentionPooler(self.hidden_dim)
        self.spa_pooler = TextAttentionPooler(self.hidden_dim)
        self.attr_pooler = TextAttentionPooler(self.hidden_dim)
        

        self.spatial_relation_encoder = StructuredPositionPointerEncoder(self.hidden_dim, num_pointers=4)

        # [约束 1] 空间定位头: 输入 Spa 向量 -> 预测 [cx, cy, w, h]
        self.spa_regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4),
            nn.Sigmoid() # 归一化坐标 0~1
        )
        
        # [约束 2] 属性门控头: 输入 Attr 向量 -> 预测是否包含属性 (0~1)
        self.attr_gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _load_weights(self, model_path, raw_model):
        weight_path = os.path.join(model_path, 'pytorch_model.bin')
        if not os.path.exists(weight_path):
            print(f"⚠️ Warning: Weights not found at {weight_path}")
            return

        print(f"Loading weights from {weight_path}...")
        hf_state_dict = torch.load(weight_path, map_location='cpu')
        
        clean_state = {}
        for k, v in hf_state_dict.items():
            if k.startswith('deberta.'):
                clean_state[k.replace('deberta.', '', 1)] = v
            elif 'lm_predictions' not in k:
                clean_state[k] = v
        
        raw_model.load_state_dict(clean_state, strict=False)
        print("✅ Pretrained weights loaded.")

    def get_rel_pos(self):
        rel_embed = self.rel_embeddings.weight
        if self.rel_layer_norm is not None:
            rel_embed = self.rel_layer_norm(rel_embed)
        return rel_embed

    def forward(self, text_input):
        device = self.embeddings.word_embeddings.weight.device

        # --- Input Processing ---
        if isinstance(text_input, list) and isinstance(text_input[0], str):
            inputs = self.tokenizer(text_input, padding='max_length', truncation=True, 
                                    max_length=self.max_len, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            mask = inputs['attention_mask'].to(device)
        elif isinstance(text_input, (list, tuple)) and len(text_input) == 2:
            input_ids = text_input[0].to(device)
            mask = text_input[1].to(device)
        else:
            raise ValueError(f"Unsupported input format: {type(text_input)}")

        # 1. Embeddings
        embedding_output = self.embeddings(input_ids.long(), mask)
        hidden_states = embedding_output['embeddings']
        
        # 2. Mask Handling
        if mask.dim() == 2:
            extended_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.to(dtype=hidden_states.dtype) 
            extended_mask = (1.0 - extended_mask) * -10000.0 # 这里是 -10000.0，是安全的
        else:
            extended_mask = mask

        # 3. Get Rel Embeddings
        rel_embed = self.get_rel_pos()
        
        # 4. Shared Encoder Forward (Base Layers)
        for layer in self.shared_encoder:
            hidden_states = layer(
                hidden_states, 
                extended_mask, 
                return_att=False, 
                rel_embeddings=rel_embed, 
                branch_type="normal"
            )
            
        # 5. Tri-Stream Forward (Branch Layers)
        
        # --- A. Semantic Branch (类别/主体) ---
        sem_feat = hidden_states
        for idx, layer in enumerate(self.semantic_encoder):
            sem_feat = layer(
                sem_feat, extended_mask, return_att=False, 
                rel_embeddings=rel_embed, branch_type="semantic", 
                layer_id=idx, total_layers=self.total_layers//2
            )
            
        # --- B. Spatial Branch (位置/关系) ---
        spa_feat = hidden_states
        for idx, layer in enumerate(self.spatial_encoder):
            spa_feat = layer(
                spa_feat, extended_mask, return_att=False, 
                rel_embeddings=rel_embed, branch_type="spatial", 
                layer_id=idx, total_layers=self.total_layers//2
            )
            
        # --- C. Attribute Branch (特征/修饰) ---
        attr_feat = hidden_states
        for idx, layer in enumerate(self.attribute_encoder):
            attr_feat = layer(
                attr_feat, extended_mask, return_att=False, 
                rel_embeddings=rel_embed, branch_type="attribute", 
                layer_id=idx, total_layers=self.total_layers//2
            )

        sem_feat = self.sem_adapter(sem_feat)
        spa_feat = self.spa_adapter(spa_feat)
        attr_feat = self.attr_adapter(attr_feat)

        # 6. Task Specific Projections
        # 获取句子级的向量表示
        sem_vec, _ = self.sem_pooler(sem_feat, mask)
        spa_vec, _ = self.spa_pooler(spa_feat, mask)
        attr_vec, _ = self.attr_pooler(attr_feat, mask)
        
        # 计算辅助任务输出
        pred_box_from_text = self.spa_regressor(spa_vec) # [B, 4]
        pred_attr_gate = self.attr_gate_head(attr_vec)   # [B, 1]

        relation_out = self.spatial_relation_encoder(spa_feat, mask)

        return {
            'sem': sem_feat,
            'spa': spa_feat,
            'attr': attr_feat,
            'sem_vec': sem_vec,
            'spa_vec': spa_vec,
            'attr_vec': attr_vec,
            'pred_box': pred_box_from_text,
            'gate': pred_attr_gate,
            'mask': mask,
            'relative_position': relation_out['relative_position'],
            'relation_tokens': relation_out['relation_tokens'],
            'relation_vec': relation_out['relation_vec'],
            'template_logits': relation_out['template_logits'],
            'template_attn': relation_out['template_attn'],
            'template_vec': relation_out['template_vec']
        }
