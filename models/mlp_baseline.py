"""
mlp_baseline.py
===============
MLP Baseline Surrogate Model for Permutation-Based Reliability Prediction

模型架構
--------
- 輸入：排列 π (K 維) + 需求 d (K 維) → flatten 成 2K 維向量
- 多層全連接網路 + BatchNorm + GELU + Dropout
- 輸出：predicted R_joint (scalar)

作為 Transformer+MLP 的 baseline 對照組。
MLP 無法自然捕捉排列中的位置與交互資訊（如 pair 之間的競爭關係），
只能靠暴力學習 2K 維輸入到輸出的映射。

公平對比設計：
- 預設 hidden_dims=[256, 256, 128]，參數量 ≈ 105K
- 與 Transformer+MLP（d_model=64, num_layers=3）的 ≈ 106K 相當
- 確保性能差異來自模型結構（self-attention vs flat），而非參數量

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


class PermutationMLP(nn.Module):
    """
    MLP baseline for predicting R_joint from (π, d).

    參數
    ----
    K : int — pair 數
    hidden_dims : list of int — 各隱藏層維度
    dropout : float — dropout rate
    """

    def __init__(self, K, hidden_dims=None, dropout=0.1):
        super().__init__()
        self.K = K

        if hidden_dims is None:
            # 預設 [256, 256, 128]：參數量 ≈ 105K，
            # 與 Transformer+MLP (d_model=64, 3 layers) 的 ≈ 106K 相當
            hidden_dims = [256, 256, 128]

        # 輸入維度：π (K 維 float) + d (K 維 float) = 2K
        input_dim = 2 * K
        layers = []

        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        # 輸出層
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, perm, demand):
        """
        前向傳播。

        參數
        ----
        perm : torch.LongTensor, shape (batch, K)
            排列矩陣（會轉成 float 作為 input feature）。
        demand : torch.FloatTensor, shape (batch, K)
            需求向量。

        回傳
        ----
        pred : torch.FloatTensor, shape (batch,)
            predicted R_joint。
        """
        # π 轉 float，歸一化到 [0, 1]
        perm_float = perm.float() / (self.K - 1) if self.K > 1 else perm.float()

        # 拼接 [π_normalized, d]
        x = torch.cat([perm_float, demand], dim=-1)  # (batch, 2K)
        pred = self.network(x).squeeze(-1)  # (batch,)

        return pred


# ============================================================
# 快速測試
# ============================================================

if __name__ == '__main__':
    K = 10
    batch = 32
    model = PermutationMLP(K=K)

    perm = torch.stack([torch.randperm(K) for _ in range(batch)])
    demand = torch.randint(1, 50, (batch, K)).float()

    pred = model(perm, demand)
    print(f"Input: perm {perm.shape}, demand {demand.shape}")
    print(f"Output: pred {pred.shape}")
    print(f"Sample predictions: {pred[:5].detach().numpy()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
