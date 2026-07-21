"""
train_surrogate.py
==================
Stage 2: Surrogate Model Training（對應論文 §3.2 — Surrogate model construction）

統一的訓練腳本，支援 Transformer 和 MLP 兩種模型。
讀取 Stage 1 生成的 .npz 資料，訓練 surrogate model 預測 R_joint。

【論文定位】
- 輸入：(π, d) — 排列 π ∈ S_K 與需求向量 d ∈ ℝ^K
- 輸出：predicted R_joint^(π)(d) ∈ [0, 1]（regression）
- Proposed model: Transformer+MLP（§3.2.1）
- Baseline model: MLP（§3.2.2）
- 公平對比：兩者參數量 ≈ 105K，差異純粹來自模型結構（self-attention vs flat）

用法
----
python train_surrogate.py --data_dir code/data/K3_psi5000 --model transformer
python train_surrogate.py --data_dir code/data/K3_psi5000 --model mlp

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


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
import argparse
import json
from scipy.stats import spearmanr

# 載入模型定義（§3.2.1 Transformer encoder / §3.2.2 MLP baseline）
from models.transformer_surrogate import PermutationTransformer
from models.mlp_baseline import PermutationMLP


# ============================================================
# Dataset（Stage 1 產出的訓練資料）
# ============================================================

class PermutationDataset(Dataset):
    """
    PyTorch Dataset：(π, d) → R_joint

    【資料格式】（對應論文 §3.1 — MC simulation-based data generation）
    每筆訓練資料包含：
    - perm (π)  : shape (K,), int — 排列（pair 的優先順序）
    - demand (d): shape (K,), float — 需求向量（各 pair 的需求量）
    - R_joint   : scalar, float — 聯合可靠度（MC 估計值）

    【正規化策略】
    - π: 保持整數（Transformer 的 embedding 需要 int index 作為 token ID）
    - d: 除以 d_max 歸一化到 [0, 1]（讓不同 pair 的需求量可比較）
    - R_joint: 保持原值（已在 [0, 1] 範圍內，不需正規化）
    """

    def __init__(self, npz_path, d_max=None):
        """
        參數
        ----
        npz_path : str — .npz 檔案路徑（Stage 1 的 data_generator.py 產出）
        d_max : np.ndarray or None — 各 pair 的 d 最大值（用於正規化）
            如果 None，從資料中自動計算（通常只在 training set 使用）
            val/test 應使用 training set 的 d_max 以保持一致
        """
        data = np.load(npz_path)
        self.perms = data['perms'].astype(np.int64)        # shape (N, K)：排列 π
        self.demands = data['demands'].astype(np.float32)  # shape (N, K)：需求 d
        self.R_joint = data['R_joint'].astype(np.float32)  # shape (N,)：聯合可靠度

        # --- Demand 正規化：d_normalized = d / d_max ---
        # d_max 從 training set 計算，val/test 共用同一組 d_max
        if d_max is None:
            self.d_max = self.demands.max(axis=0)       # 各 pair 的最大需求
            self.d_max = np.maximum(self.d_max, 1.0)    # 避免除以 0
        else:
            self.d_max = d_max.astype(np.float32)

        self.demands_normalized = self.demands / self.d_max  # shape (N, K)，值在 [0, 1]

    def __len__(self):
        return len(self.R_joint)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.perms[idx]),              # π: shape (K,), int64
            torch.from_numpy(self.demands_normalized[idx]), # d/d_max: shape (K,), float32
            torch.tensor(self.R_joint[idx]),                # R_joint: scalar, float32
        )


# ============================================================
# 評估指標（對應論文 §4.2 — Surrogate prediction accuracy）
# ============================================================

def evaluate_model(model, dataloader, device):
    """
    在 dataloader 上評估模型的預測品質。

    【評估指標】（論文 Table 2 使用的四項指標）
    - MAE: Mean Absolute Error — 預測誤差的平均絕對值
    - MSE: Mean Squared Error — 預測誤差的均方值
    - R²: Coefficient of Determination — 模型解釋力（1.0 = 完美）
    - Spearman: Spearman Rank Correlation — 排序保持能力
      （對 GA 最重要：surrogate 不需要精確預測值，但必須保持排序一致性）

    回傳
    ----
    metrics : dict — 包含 MAE, MSE, R², Spearman, preds, targets
    """
    model.eval()  # 切換到評估模式（關閉 dropout）
    all_preds = []
    all_targets = []

    with torch.no_grad():  # 推論時不計算梯度
        for perm, demand, target in dataloader:
            perm = perm.to(device)      # π → GPU/MPS
            demand = demand.to(device)  # d → GPU/MPS
            # 前向傳播：model(π, d) → predicted R_joint
            pred = model(perm, demand)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())

    preds = np.concatenate(all_preds)     # shape (N,)
    targets = np.concatenate(all_targets) # shape (N,)

    # --- MAE: Mean Absolute Error ---
    mae = np.mean(np.abs(preds - targets))

    # --- MSE: Mean Squared Error ---
    mse = np.mean((preds - targets) ** 2)

    # --- R²: Coefficient of Determination ---
    # R² = 1 - SS_res / SS_tot
    # SS_res = Σ(y - ŷ)², SS_tot = Σ(y - ȳ)²
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # --- Spearman Rank Correlation ---
    # 衡量預測值的排序是否與真值排序一致
    # 對 GA 的 fitness ranking 至關重要（§3.4.3）
    if len(np.unique(targets)) > 1 and len(np.unique(preds)) > 1:
        spearman_corr, _ = spearmanr(targets, preds)
    else:
        spearman_corr = 0.0

    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'R2': float(r2),
        'Spearman': float(spearman_corr),
        'preds': preds,
        'targets': targets,
    }


# ============================================================
# 訓練主函式
# ============================================================

def train(model, train_loader, val_loader, device,
          epochs=200, lr=1e-3, weight_decay=1e-4,
          patience=30, min_delta=1e-6):
    """
    訓練 surrogate model（對應論文 §3.2 的訓練流程）

    【訓練策略】
    - Optimizer: AdamW（Adam + weight decay，防止過擬合）
    - Scheduler: CosineAnnealingLR（學習率從 lr 逐漸降到接近 0）
    - Loss: MSE Loss（regression task）
    - Gradient Clipping: max_norm=1.0（穩定訓練）
    - Early Stopping: 若 val MAE 連續 patience 個 epoch 無改善則停止

    參數
    ----
    patience : int — early stopping 的耐心（多少 epoch 沒改善就停）
    min_delta : float — 認定「改善」的最小 MAE 下降

    回傳
    ----
    history : dict — 訓練歷史 (train_loss, val_mae, val_r2, val_spearman per epoch)
    best_state : dict — 最佳 model 的 state_dict（val MAE 最低時）
    """
    # --- 優化器與排程器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()  # L = (1/N) Σ(R_joint - R̂_joint)²

    # --- 訓練歷史紀錄 ---
    history = {
        'train_loss': [],       # 每 epoch 的訓練 loss
        'val_mae': [],          # 每 epoch 的驗證 MAE
        'val_r2': [],           # 每 epoch 的驗證 R²
        'val_spearman': [],     # 每 epoch 的驗證 Spearman
    }

    # --- Early Stopping 狀態 ---
    best_val_mae = float('inf')  # 目前最佳驗證 MAE
    best_state = None            # 最佳模型的參數快照
    patience_counter = 0         # 連續未改善的 epoch 計數

    for epoch in range(epochs):
        # === Training Phase ===
        model.train()  # 開啟 dropout
        epoch_loss = 0.0
        n_batches = 0

        for perm, demand, target in train_loader:
            # 資料搬到 device
            perm = perm.to(device)      # π: (batch, K)
            demand = demand.to(device)  # d: (batch, K)
            target = target.to(device)  # R_joint: (batch,)

            optimizer.zero_grad()
            # 前向傳播：model(π, d) → predicted R̂_joint
            pred = model(perm, demand)
            # 計算 MSE loss
            loss = criterion(pred, target)
            # 反向傳播
            loss.backward()

            # 梯度裁剪：防止梯度爆炸（Transformer 訓練常見技巧）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新參數
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # 更新學習率（CosineAnnealing）
        scheduler.step()
        avg_loss = epoch_loss / n_batches

        # === Validation Phase ===
        val_metrics = evaluate_model(model, val_loader, device)

        # 紀錄歷史
        history['train_loss'].append(avg_loss)
        history['val_mae'].append(val_metrics['MAE'])
        history['val_r2'].append(val_metrics['R2'])
        history['val_spearman'].append(val_metrics['Spearman'])

        # === Early Stopping 檢查 ===
        if val_metrics['MAE'] < best_val_mae - min_delta:
            # 有改善：更新最佳模型，重置計數器
            best_val_mae = val_metrics['MAE']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            # 無改善：計數器 +1
            patience_counter += 1

        # --- 每 20 epoch 印一次訓練狀態 ---
        if (epoch + 1) % 20 == 0 or epoch == 0 or patience_counter >= patience:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:>4d}/{epochs} | "
                  f"Loss={avg_loss:.6f} | "
                  f"Val MAE={val_metrics['MAE']:.6f} | "
                  f"Val R²={val_metrics['R2']:.4f} | "
                  f"Spearman={val_metrics['Spearman']:.4f} | "
                  f"LR={current_lr:.2e} | "
                  f"{'★' if patience_counter == 0 else f'patience {patience_counter}/{patience}'}")

        # 觸發 Early Stopping
        if patience_counter >= patience:
            print(f"  [Early Stopping] Epoch {epoch+1}, best Val MAE={best_val_mae:.6f}")
            break

    # 載入最佳模型（val MAE 最低時的參數）
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_state


# ============================================================
# CLI 主程式
# ============================================================

def main():
    # --- 命令列參數 ---
    parser = argparse.ArgumentParser(description='Stage 2: Train Surrogate Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Stage 1 data 資料夾（包含 train.npz, val.npz, test.npz）')
    parser.add_argument('--model', type=str, default='transformer',
                        choices=['transformer', 'mlp'],
                        help='模型類型：transformer (§3.2.1) 或 mlp (§3.2.2)')
    parser.add_argument('--epochs', type=int, default=200, help='最大訓練 epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate（初始值）')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Transformer hidden dim（§3.2.1 的 d_model）')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Transformer attention heads（multi-head 數量）')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Transformer encoder layers（堆疊層數）')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu/mps)')
    parser.add_argument('--save_dir', type=str, default=None, help='模型儲存路徑')
    args = parser.parse_args()

    # --- Device 自動偵測 ---
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')        # NVIDIA GPU（RTX 4090）
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')          # Apple Silicon GPU
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # 模型顯示名稱（論文用）
    model_display = 'Transformer+MLP' if args.model == 'transformer' else 'MLP'

    print("=" * 60)
    print(f"  Stage 2: Train Surrogate ({model_display})")
    print(f"  Data: {args.data_dir}")
    print(f"  Device: {device}")
    print("=" * 60)

    # --- 載入資料（Stage 1 的 .npz 檔）---
    print("\n[載入資料]")
    train_ds = PermutationDataset(os.path.join(args.data_dir, 'train.npz'))
    d_max = train_ds.d_max  # 用 training set 的 d_max 正規化（val/test 共用）

    val_ds = PermutationDataset(os.path.join(args.data_dir, 'val.npz'), d_max=d_max)
    test_ds = PermutationDataset(os.path.join(args.data_dir, 'test.npz'), d_max=d_max)

    K = train_ds.perms.shape[1]  # K: pair 數（從資料維度推斷）
    print(f"  K={K}, d_max={d_max}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  R_joint — train: mean={train_ds.R_joint.mean():.6f}, "
          f"std={train_ds.R_joint.std():.6f}")

    # --- 建立 DataLoader ---
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == 'cuda'))

    # --- 建立模型（§3.2.1 Transformer 或 §3.2.2 MLP）---
    print(f"\n[建立模型: {args.model.upper()}]")
    if args.model == 'transformer':
        # §3.2.1: Transformer+MLP — Self-Attention 捕捉 pair 間的交互作用
        # token = pair_embedding + positional_embedding + demand_projection
        # → Transformer Encoder (self-attention)
        # → Global Average Pooling (d_model 維)
        # → MLP head → R_joint
        model = PermutationTransformer(
            K=K,                                 # pair 數（決定序列長度）
            d_model=args.d_model,                # hidden dimension
            nhead=args.nhead,                    # multi-head attention 頭數
            num_layers=args.num_layers,          # encoder 層數
            dim_feedforward=args.d_model * 2,    # FFN 中間層維度
            dropout=0.1,
        )
    else:
        # §3.2.2: MLP baseline — π + d flatten 成 2K 維向量 → 3 層 MLP [256,256,128]
        model = PermutationMLP(K=K, dropout=0.1)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")  # 兩者皆 ≈ 105K 參數

    # --- 訓練 ---
    print(f"\n[開始訓練] epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    t_start = time.perf_counter()

    history, best_state = train(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
    )

    train_time = time.perf_counter() - t_start
    print(f"  訓練完成，耗時 {train_time:.1f}s")

    # --- 測試集評估（最終報告的指標，對應論文 Table 2）---
    print(f"\n[測試集評估]")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"  MAE:      {test_metrics['MAE']:.6f}")
    print(f"  MSE:      {test_metrics['MSE']:.8f}")
    print(f"  R²:       {test_metrics['R2']:.4f}")
    print(f"  Spearman: {test_metrics['Spearman']:.4f}")

    # --- 儲存模型（.pt 檔，供 Stage 3 GA 載入）---
    if args.save_dir is None:
        args.save_dir = os.path.join(args.data_dir, 'models')

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f'{args.model}_best.pt')
    torch.save({
        'model_state_dict': best_state,  # 最佳模型參數
        'model_type': args.model,        # 'transformer' 或 'mlp'
        'K': K,                          # pair 數
        'd_max': d_max,                  # demand 正規化基準（GA 需要）
        'test_metrics': {k: v for k, v in test_metrics.items()
                         if k not in ('preds', 'targets')},  # 測試指標
        'history': history,              # 訓練歷史
        'args': vars(args),              # 超參數（重建模型用）
    }, model_path)
    print(f"\n  [存檔] {model_path}")

    # --- 儲存測試結果（JSON，方便比較不同模型）---
    results_path = os.path.join(args.save_dir, f'{args.model}_results.json')
    results = {
        'model': args.model,
        'model_display': model_display,
        'K': K,
        'n_params': n_params,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
        'train_time_sec': train_time,
        'epochs_trained': len(history['train_loss']),
        'test_MAE': test_metrics['MAE'],
        'test_MSE': test_metrics['MSE'],
        'test_R2': test_metrics['R2'],
        'test_Spearman': test_metrics['Spearman'],
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [存檔] {results_path}")

    print(f"\n{'='*60}")
    print(f"  完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
