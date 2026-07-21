"""
transformer_surrogate.py
========================
Transformer + MLP Surrogate Model for Permutation-Based Reliability Prediction

模型架構（Proposed Method: Transformer + MLP）
----------------------------------------------
- 輸入：排列 π (K 維整數序列) + 需求向量 d (K 維連續值)
- Transformer Encoder：捕捉排列中各 pair 之間的交互作用（self-attention）
- MLP Regression Head：Global avg pooling → Linear → GELU → Dropout → Linear → R_joint

與 MLP baseline 的差異：
- Transformer+MLP 用 self-attention 建模「pair 之間的競爭關係」
  （如 pair i 先分配容量 → pair j 可用容量減少）
- MLP baseline 只能靠 flat vector 暴力學習
- 兩者參數量相當（~105K vs ~106K），差異純粹來自模型結構

關鍵設計
--------
1. Pair Embedding：將 pair index (0~K-1) 映射成 d_model 維向量
   - 這不是 positional encoding，而是「這是哪個 pair」的身份
2. Positional Encoding：表示「在排列中排第幾」（優先順序）
   - learnable positional embedding
3. Demand Feature：d_k 經過一層 linear 映射後加到 token 上
   - 讓模型知道每個 pair 的需求大小
4. Self-Attention：學習「pair i 排在 pair j 前面」對 R_joint 的影響

"""

# --- path bootstrap (restructured package) ---
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('', 'core', 'models', 'instances', 'baselines', 'experiments', 'figures_code'):
    _p = _os.path.join(ROOT, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
HERE = ROOT   # data/ and figures/ live at the package root after restructuring
CODE = ROOT


import torch
import torch.nn as nn
import math


class PermutationTransformer(nn.Module):
    """
    Transformer Encoder for predicting R_joint from (π, d).

    參數
    ----
    K : int — pair 數（序列長度）
    d_model : int — Transformer hidden dimension
    nhead : int — multi-head attention heads 數
    num_layers : int — Transformer encoder layers 數
    dim_feedforward : int — feedforward 層維度
    dropout : float — dropout rate
    """

    def __init__(self, K, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.K = K
        self.d_model = d_model

        # --- Pair Embedding：pair index → d_model 維 ---
        # π = [3, 0, 1, 2] 時，position 0 的 token 用 pair_emb[3]
        self.pair_embedding = nn.Embedding(K, d_model)

        # --- Positional Embedding：position in permutation → d_model 維 ---
        # position 0 = 最高優先權，position K-1 = 最低優先權
        self.pos_embedding = nn.Embedding(K, d_model)

        # --- Demand Projection：d_k (scalar) → d_model 維 ---
        self.demand_proj = nn.Linear(1, d_model)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # input shape: (batch, seq_len, d_model)
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # --- Regression Head ---
        # 用 [CLS]-like 的 global pooling + MLP
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, perm, demand):
        """
        前向傳播。

        參數
        ----
        perm : torch.LongTensor, shape (batch, K)
            排列矩陣。perm[b, pos] = 排在第 pos 位的 pair index。
        demand : torch.FloatTensor, shape (batch, K)
            需求向量。demand[b, k] = 第 k 組 pair 的需求值。

        回傳
        ----
        pred : torch.FloatTensor, shape (batch,)
            predicted R_joint。
        """
        batch_size = perm.shape[0]

        # 1. Pair embedding：根據排列中每個位置是哪個 pair
        #    pair_emb[b, pos] = embedding of pair perm[b, pos]
        pair_emb = self.pair_embedding(perm)  # (batch, K, d_model)

        # 2. Positional embedding：每個位置的優先權資訊
        positions = torch.arange(self.K, device=perm.device)
        pos_emb = self.pos_embedding(positions)  # (K, d_model)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, K, d_model)

        # 3. Demand feature：取出每個位置對應的 pair 的 demand 值
        #    demand_at_pos[b, pos] = demand[b, perm[b, pos]]
        demand_at_pos = torch.gather(demand, 1, perm)  # (batch, K)
        demand_at_pos = demand_at_pos.unsqueeze(-1)  # (batch, K, 1)
        demand_emb = self.demand_proj(demand_at_pos)  # (batch, K, d_model)

        # 4. 合併：pair identity + position + demand
        x = pair_emb + pos_emb + demand_emb  # (batch, K, d_model)

        # 5. Transformer Encoder
        x = self.transformer_encoder(x)  # (batch, K, d_model)

        # 6. Global average pooling → regression
        x = x.mean(dim=1)  # (batch, d_model)
        pred = self.regression_head(x).squeeze(-1)  # (batch,)

        return pred


# ============================================================
# 快速測試
# ============================================================

if __name__ == '__main__':
    K = 10
    batch = 32
    model = PermutationTransformer(K=K, d_model=64, nhead=4, num_layers=3)

    # 隨機 input
    perm = torch.stack([torch.randperm(K) for _ in range(batch)])
    demand = torch.randint(1, 50, (batch, K)).float()

    pred = model(perm, demand)
    print(f"Input: perm {perm.shape}, demand {demand.shape}")
    print(f"Output: pred {pred.shape}")
    print(f"Sample predictions: {pred[:5].detach().numpy()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
