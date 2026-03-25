"""
lstm_model.py
-------------
PyTorch LSTM，把球隊近 10 場的得失分序列壓縮成「球隊狀態向量」，
再接全連接層輸出預測總分。

架構：
  [home_seq: 10×3] → LSTM → hidden → ┐
                                       FC → 預測總分
  [away_seq: 10×3] → LSTM → hidden → ┘ + 賽前特徵

輸入序列特徵（每場 3 個值）：
  scored, allowed, total

使用方式：
  python src/models/lstm_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 設定 ──────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

FEAT_DIR   = Path(__file__).parent.parent.parent / CFG["data"]["features_dir"]
MODEL_DIR  = Path(__file__).parent.parent.parent / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LSTM_CFG     = CFG["model"]["lstm"]
TEST_SEASONS = CFG["model"]["test_seasons"]
VAL_SEASONS  = CFG["model"]["val_seasons"]
SEED         = CFG["model"]["random_seed"]

SEQ_LEN      = LSTM_CFG["seq_len"]        # 10
HIDDEN_SIZE  = LSTM_CFG["hidden_size"]    # 64
NUM_LAYERS   = LSTM_CFG["num_layers"]     # 2
DROPOUT      = LSTM_CFG["dropout"]        # 0.3
EPOCHS       = LSTM_CFG["epochs"]         # 100
BATCH_SIZE   = LSTM_CFG["batch_size"]     # 64
LR           = LSTM_CFG["lr"]             # 0.001

SEQ_FEATURES = ["scored", "allowed", "total"]  # 每個時步 3 個值
INPUT_SIZE   = len(SEQ_FEATURES)

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────

class NBADataset(Dataset):
    """
    每個樣本包含：
      home_seq : (SEQ_LEN, 3)  主場球隊近期序列
      away_seq : (SEQ_LEN, 3)  客場球隊近期序列
      meta     : (n_meta,)     賽前純量特徵
      target   : float         總得分
    """

    def __init__(self, df: pd.DataFrame, meta_cols: list[str],
                 scaler_meta: StandardScaler = None, fit_scaler: bool = False):
        self.samples = []

        # 建立每支球隊的歷史序列 lookup
        # long format: {team: [(scored, allowed, total), ...]}
        team_history: dict[str, list] = {}

        long = []
        for _, row in df.sort_values("game_date").iterrows():
            for side in ["home", "away"]:
                team = row[f"{side}_team"]
                scored  = row[f"{side}_total"]
                allowed = row["away_total" if side == "home" else "home_total"]
                total   = row["total_score"]
                long.append({
                    "team": team,
                    "game_date": row["game_date"],
                    "game_id":   row["game_id"],
                    "scored":  scored,
                    "allowed": allowed,
                    "total":   total,
                })
        long_df = pd.DataFrame(long).sort_values("game_date")

        def get_seq(team: str, before_date) -> np.ndarray | None:
            hist = long_df[(long_df["team"] == team) &
                           (long_df["game_date"] < before_date)].tail(SEQ_LEN)
            if len(hist) < 3:
                return None
            seq = hist[SEQ_FEATURES].values.astype(np.float32)
            # 不足 SEQ_LEN 則補零 padding
            if len(seq) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(seq), INPUT_SIZE), dtype=np.float32)
                seq = np.vstack([pad, seq])
            return seq

        # 準備 meta 特徵縮放
        meta_vals = df[meta_cols].fillna(0).values.astype(np.float32)
        if fit_scaler:
            scaler_meta = StandardScaler()
            meta_vals = scaler_meta.fit_transform(meta_vals)
        elif scaler_meta:
            meta_vals = scaler_meta.transform(meta_vals)
        self.scaler_meta = scaler_meta

        for i, (_, row) in enumerate(df.sort_values("game_date").iterrows()):
            home_seq = get_seq(row["home_team"], row["game_date"])
            away_seq = get_seq(row["away_team"], row["game_date"])

            if home_seq is None or away_seq is None:
                continue

            self.samples.append({
                "home_seq": torch.tensor(home_seq),
                "away_seq": torch.tensor(away_seq),
                "meta":     torch.tensor(meta_vals[i]),
                "target":   torch.tensor(float(row["total_score"])),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── 模型架構 ──────────────────────────────────────────────────

class NBALSTMModel(nn.Module):
    def __init__(self, n_meta: int):
        super().__init__()
        # 主客場共享同一個 LSTM 權重（siamese style）
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(DROPOUT)

        # 全連接層：LSTM hidden × 2 (home+away) + meta 特徵
        fc_input = HIDDEN_SIZE * 2 + n_meta
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (batch, SEQ_LEN, INPUT_SIZE) → (batch, HIDDEN_SIZE)"""
        _, (h_n, _) = self.lstm(seq)
        return self.dropout(h_n[-1])   # 取最後一層 hidden state

    def forward(self, home_seq, away_seq, meta):
        home_enc = self.encode(home_seq)    # (B, HIDDEN_SIZE)
        away_enc = self.encode(away_seq)    # (B, HIDDEN_SIZE)
        combined = torch.cat([home_enc, away_enc, meta], dim=1)
        return self.fc(combined).squeeze(1)


# ── 訓練 ──────────────────────────────────────────────────────

META_COLS = [
    "home_avg_scored_10", "home_avg_allowed_10",
    "away_avg_scored_10", "away_avg_allowed_10",
    "implied_total_10",   "pace_matchup",
    "home_rest_days",     "away_rest_days",
    "home_b2b",           "away_b2b",
    "q1_total",           "h1_total",
    "q1_pace_ratio",      "h1_pace_ratio",
    "q1_home_lead",       "h1_home_lead",
]


def build_datasets(df: pd.DataFrame):
    test_mask  = df["season"].isin(TEST_SEASONS)
    val_mask   = df["season"].isin(VAL_SEASONS)
    train_mask = ~test_mask & ~val_mask

    available_meta = [c for c in META_COLS if c in df.columns]

    train_ds = NBADataset(df[train_mask], available_meta, fit_scaler=True)
    val_ds   = NBADataset(df[val_mask],   available_meta,
                          scaler_meta=train_ds.scaler_meta)
    test_ds  = NBADataset(df[test_mask],  available_meta,
                          scaler_meta=train_ds.scaler_meta)

    log.info(f"Dataset sizes — train:{len(train_ds)}  "
             f"val:{len(val_ds)}  test:{len(test_ds)}")
    return train_ds, val_ds, test_ds, available_meta


def train_model(train_ds, val_ds, n_meta: int) -> NBALSTMModel:
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = NBALSTMModel(n_meta=n_meta).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.HuberLoss()   # 比 MSE 對 outlier 更穩健

    best_val_mae = float("inf")
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(
                batch["home_seq"].to(DEVICE),
                batch["away_seq"].to(DEVICE),
                batch["meta"].to(DEVICE),
            )
            loss = criterion(pred, batch["target"].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── validate ──
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                p = model(
                    batch["home_seq"].to(DEVICE),
                    batch["away_seq"].to(DEVICE),
                    batch["meta"].to(DEVICE),
                )
                preds.extend(p.cpu().numpy())
                targets.extend(batch["target"].numpy())

        val_mae = mean_absolute_error(targets, preds)
        scheduler.step(val_mae)

        if epoch % 10 == 0:
            log.info(f"Epoch {epoch:3d}/{EPOCHS}  val MAE: {val_mae:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    log.info(f"LSTM best val MAE: {best_val_mae:.3f}")
    return model


# ── 推論（支援單筆即時預測）────────────────────────────────────

def predict_single(model: NBALSTMModel,
                   home_seq: np.ndarray,
                   away_seq: np.ndarray,
                   meta: np.ndarray) -> float:
    """
    單筆推論，用於 predictor.py 即時預測。
    """
    model.eval()
    with torch.no_grad():
        hs = torch.tensor(home_seq).unsqueeze(0).to(DEVICE)
        as_ = torch.tensor(away_seq).unsqueeze(0).to(DEVICE)
        m  = torch.tensor(meta.astype(np.float32)).unsqueeze(0).to(DEVICE)
        return float(model(hs, as_, m).cpu().item())


# ── 存讀 ──────────────────────────────────────────────────────

def save(model: NBALSTMModel, scaler, meta_cols: list, name: str = "lstm"):
    torch.save(model.state_dict(), MODEL_DIR / f"{name}.pt")
    with open(MODEL_DIR / f"{name}_meta.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "meta_cols": meta_cols,
                     "n_meta": len(meta_cols)}, f)
    log.info(f"LSTM saved → {MODEL_DIR / name}.pt")


def load(name: str = "lstm") -> tuple:
    with open(MODEL_DIR / f"{name}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    model = NBALSTMModel(n_meta=meta["n_meta"]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / f"{name}.pt",
                                     map_location=DEVICE))
    model.eval()
    return model, meta["scaler"], meta["meta_cols"]


# ── 進入點 ────────────────────────────────────────────────────

def main():
    feat_path = FEAT_DIR / "features.csv"
    df = pd.read_csv(feat_path, parse_dates=["game_date"])

    train_ds, val_ds, test_ds, meta_cols = build_datasets(df)
    model = train_model(train_ds, val_ds, n_meta=len(meta_cols))

    # 測試集評估
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            p = model(batch["home_seq"].to(DEVICE),
                      batch["away_seq"].to(DEVICE),
                      batch["meta"].to(DEVICE))
            preds.extend(p.cpu().numpy())
            targets.extend(batch["target"].numpy())

    test_mae = mean_absolute_error(targets, preds)
    log.info(f"LSTM TEST MAE: {test_mae:.3f}")

    save(model, train_ds.scaler_meta, meta_cols)


if __name__ == "__main__":
    main()
