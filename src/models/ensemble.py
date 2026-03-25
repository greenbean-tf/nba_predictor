"""
ensemble.py
-----------
把 XGBoost、LightGBM、LSTM 三個模型的預測加權平均，
輸出最終總得分預測，再傳入 output.py 轉成 over/under 信號。

權重（來自 config.yaml）：
  XGBoost  : 40%
  LightGBM : 40%
  LSTM     : 20%

使用方式：
  python src/models/ensemble.py          # 訓練三個模型 + 回測
  python src/models/ensemble.py --eval   # 只評估（模型已存在）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import yaml

# ── 設定 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

FEAT_DIR  = Path(__file__).parent.parent.parent / CFG["data"]["features_dir"]
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEST_SEASONS = CFG["model"]["test_seasons"]
VAL_SEASONS  = CFG["model"]["val_seasons"]
ENS_CFG      = CFG["model"]["ensemble"]

W_XGB  = ENS_CFG["xgb_weight"]    # 0.40
W_LGB  = ENS_CFG["lgb_weight"]    # 0.40
W_LSTM = ENS_CFG["lstm_weight"]   # 0.20


# ── 載入各模型預測 ────────────────────────────────────────────

def get_predictions(df: pd.DataFrame,
                    split: str = "test") -> pd.DataFrame:
    """
    對指定資料集（val / test）跑三個模型，回傳預測 DataFrame。

    回傳欄位：
      game_id, game_date, home_team, away_team,
      total_score,
      pred_xgb, pred_lgb, pred_lstm,
      pred_ensemble
    """
    # 延遲 import，避免循環依賴
    import src.models.xgb_model  as xgb_mod
    import src.models.lgb_model  as lgb_mod
    import src.models.lstm_model as lstm_mod

    # 載入模型
    xgb_model  = xgb_mod.load()
    lgb_model  = lgb_mod.load()
    lstm_model, lstm_scaler, lstm_meta_cols = lstm_mod.load()

    # 切資料
    if split == "test":
        mask = df["season"].isin(TEST_SEASONS)
    else:
        mask = df["season"].isin(VAL_SEASONS)

    sub = df[mask].copy()

    # XGBoost
    feat_cols = xgb_mod.get_feature_cols(df)
    feat_cols = [c for c in feat_cols if c in sub.columns]
    pred_xgb  = xgb_model.predict(sub[feat_cols])

    # LightGBM
    feat_cols_lgb = lgb_mod.get_feature_cols(df)
    feat_cols_lgb = [c for c in feat_cols_lgb if c in sub.columns]
    pred_lgb  = lgb_model.predict(sub[feat_cols_lgb])

    # LSTM（批次推論）
    pred_lstm = _batch_predict_lstm(lstm_model, lstm_scaler,
                                    lstm_meta_cols, df, sub)

    # Ensemble（加權平均）
    pred_ensemble = W_XGB * pred_xgb + W_LGB * pred_lgb + W_LSTM * pred_lstm

    result = sub[["game_id", "game_date", "season", "home_team",
                  "away_team", "total_score"]].copy()
    result["pred_xgb"]      = pred_xgb
    result["pred_lgb"]      = pred_lgb
    result["pred_lstm"]     = pred_lstm
    result["pred_ensemble"] = pred_ensemble

    return result.reset_index(drop=True)


def _batch_predict_lstm(model, scaler, meta_cols: list,
                        full_df: pd.DataFrame,
                        sub: pd.DataFrame) -> np.ndarray:
    """
    對 sub 中的每場比賽執行 LSTM 推論。
    （LSTM Dataset 建構需要完整歷史，所以傳入 full_df）
    """
    import torch
    from src.models.lstm_model import NBADataset, DEVICE, BATCH_SIZE
    from torch.utils.data import DataLoader

    available_meta = [c for c in meta_cols if c in sub.columns]
    ds = NBADataset(sub, available_meta, scaler_meta=scaler)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            p = model(
                batch["home_seq"].to(DEVICE),
                batch["away_seq"].to(DEVICE),
                batch["meta"].to(DEVICE),
            )
            preds.extend(p.cpu().numpy())

    # 若 dataset 跳過了部分樣本（歷史不足），補 XGB 預測的平均值填充
    if len(preds) < len(sub):
        log.warning(f"LSTM predicted {len(preds)} / {len(sub)} samples, "
                    f"filling gaps with mean")
        mean_val = np.mean(preds) if preds else 225.0
        result = np.full(len(sub), mean_val)
        result[:len(preds)] = preds
        return result

    return np.array(preds)


# ── 評估 ──────────────────────────────────────────────────────

def evaluate(result: pd.DataFrame) -> dict:
    """計算各模型 MAE 並比較。"""
    metrics = {}
    for col in ["pred_xgb", "pred_lgb", "pred_lstm", "pred_ensemble"]:
        mae = mean_absolute_error(result["total_score"], result[col])
        metrics[col] = mae

    print(f"\n{'═'*45}")
    print(f"{'  Model Comparison':^45}")
    print(f"{'═'*45}")
    best_mae = min(metrics.values())
    for name, mae in metrics.items():
        label    = name.replace("pred_", "").upper()
        marker   = " ★" if mae == best_mae else ""
        bar      = "█" * int((1 - mae / 20) * 20)
        print(f"  {label:<12} MAE: {mae:6.3f}  {bar}{marker}")
    print(f"{'═'*45}\n")

    return metrics


# ── 即時預測（單場）─────────────────────────────────────────

def predict_live(game: dict,
                 team_history: pd.DataFrame,
                 implied_total: float) -> dict:
    """
    即時預測單場比賽總得分。
    由 predictor.py 呼叫。

    Args:
        game          : live_scraper 回傳的比賽 dict
        team_history  : 歷史特徵 DataFrame（從 features.csv 過濾）
        implied_total : 莊家開盤線

    Returns:
        {
          "pred_total":    229.3,
          "pred_std":      9.5,     ← 用三模型預測的標準差估算
          "pred_xgb":      230.1,
          "pred_lgb":      229.0,
          "pred_lstm":     228.2,
        }
    """
    import src.models.xgb_model  as xgb_mod
    import src.models.lgb_model  as lgb_mod
    import src.models.lstm_model as lstm_mod
    from src.features.manual_features import build_features_live

    # 建立即時特徵
    feats = build_features_live(game, team_history, implied_total)
    feat_df = feats.to_frame().T

    # XGB
    xgb_model = xgb_mod.load()
    xgb_cols  = [c for c in xgb_mod.get_feature_cols(team_history)
                 if c in feat_df.columns]
    p_xgb = float(xgb_model.predict(feat_df[xgb_cols])[0])

    # LGB
    lgb_model = lgb_mod.load()
    lgb_cols  = [c for c in lgb_mod.get_feature_cols(team_history)
                 if c in feat_df.columns]
    p_lgb = float(lgb_model.predict(feat_df[lgb_cols])[0])

    # LSTM
    lstm_model, scaler, meta_cols = lstm_mod.load()
    # 建立序列
    def get_team_seq(team: str) -> np.ndarray:
        mask = ((team_history["home_team"] == team) |
                (team_history["away_team"] == team))
        hist = team_history[mask].tail(lstm_mod.SEQ_LEN)
        rows = []
        for _, r in hist.iterrows():
            scored  = r["home_total"] if r["home_team"] == team else r["away_total"]
            allowed = r["away_total"] if r["home_team"] == team else r["home_total"]
            rows.append([scored, allowed, r["total_score"]])
        seq = np.array(rows, dtype=np.float32)
        if len(seq) < lstm_mod.SEQ_LEN:
            pad = np.zeros((lstm_mod.SEQ_LEN - len(seq), 3), dtype=np.float32)
            seq = np.vstack([pad, seq])
        return seq

    home_seq = get_team_seq(game["home_team"])
    away_seq = get_team_seq(game["away_team"])
    meta_vals = feats[meta_cols].fillna(0).values.reshape(1, -1).astype(np.float32)
    if scaler:
        meta_vals = scaler.transform(meta_vals)
    p_lstm = lstm_mod.predict_single(lstm_model, home_seq, away_seq, meta_vals[0])

    # Ensemble
    p_ensemble = W_XGB * p_xgb + W_LGB * p_lgb + W_LSTM * p_lstm
    pred_std   = float(np.std([p_xgb, p_lgb, p_lstm]))  # 三模型分歧 → 不確定度

    return {
        "pred_total": round(p_ensemble, 2),
        "pred_std":   round(max(pred_std, 5.0), 2),  # 最小 5 分
        "pred_xgb":   round(p_xgb,  2),
        "pred_lgb":   round(p_lgb,  2),
        "pred_lstm":  round(p_lstm, 2),
    }


# ── 進入點 ────────────────────────────────────────────────────

def main(eval_only: bool = False):
    # 優先用 features_full.csv（含 tsfresh 特徵），否則 fallback
    feat_full  = FEAT_DIR / "features_full.csv"
    feat_basic = FEAT_DIR / "features.csv"
    feat_path  = feat_full if feat_full.exists() else feat_basic
    log.info(f"Loading features from: {feat_path.name}")
    df = pd.read_csv(feat_path, parse_dates=["game_date"])

    if not eval_only:
        log.info("Training XGBoost...")
        import src.models.xgb_model as xgb_mod
        xgb_mod.main()

        log.info("Training LightGBM...")
        import src.models.lgb_model as lgb_mod
        lgb_mod.main()

        log.info("Training LSTM...")
        import src.models.lstm_model as lstm_mod
        lstm_mod.main()

    log.info("Evaluating ensemble on test set...")
    result = get_predictions(df, split="test")
    metrics = evaluate(result)

    # 儲存預測結果供 backtest 使用
    out_path = MODEL_DIR / "test_predictions.csv"
    result.to_csv(out_path, index=False)
    log.info(f"Predictions saved → {out_path}")

    return result, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Only evaluate, skip training")
    args = parser.parse_args()
    main(eval_only=args.eval)
