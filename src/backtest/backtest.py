"""
backtest.py
-----------
回測框架：載入 ensemble 預測結果，對 14 個 threshold 計算：
  - Over/Under 勝率
  - 只在高信心時下注的勝率與場次
  - 每季分析
  - ROI 估算（假設標準 -110 賠率）

使用方式：
  python src/backtest/backtest.py
  python src/backtest/backtest.py --split val
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
LOG_DIR   = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = CFG["output"]["thresholds"]
CONF_HIGH  = CFG["output"]["confidence_levels"]["high"]    # 3.0
CONF_MED   = CFG["output"]["confidence_levels"]["medium"]  # 1.5

# -110 賠率損益平衡需要 52.38% 勝率
BREAKEVEN_RATE = 110 / 210


# ── 載入預測 ──────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    path = MODEL_DIR / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ensemble.py first."
        )
    df = pd.read_csv(path, parse_dates=["game_date"])
    log.info(f"Loaded {len(df)} predictions")
    return df


# ── 核心回測邏輯 ──────────────────────────────────────────────

def backtest_threshold(df: pd.DataFrame, line: float,
                       conf_min: float = 0.0) -> dict:
    """
    對單一 threshold 做回測。

    Args:
        df       : 含 pred_ensemble 和 total_score 的 DataFrame
        line     : 莊家線（e.g. 226.5）
        conf_min : |pred - line| 需超過此值才下注

    Returns:
        dict with line, n_bets, win_rate, roi
    """
    diff     = df["pred_ensemble"] - line
    bet_over = diff > 0
    conf     = diff.abs()
    eligible = conf >= conf_min

    actual_over = df["total_score"] > line
    n_bets      = eligible.sum()

    if n_bets == 0:
        return {"line": line, "n_bets": 0, "win_rate": None, "roi": None}

    correct  = ((bet_over == actual_over) & eligible).sum()
    win_rate = correct / n_bets

    # ROI（-110 賠率）
    roi = (correct * 100 - (n_bets - correct) * 110) / (n_bets * 110)

    return {
        "line":      line,
        "n_bets":    int(n_bets),
        "n_correct": int(correct),
        "win_rate":  round(float(win_rate), 4),
        "roi":       round(float(roi), 4),
    }


def run_full_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """對 14 個 threshold × 3 個信心門檻做完整回測。"""
    rows = []
    for line in THRESHOLDS:
        for conf_label, conf_min in [("全部",   0.0),
                                     ("中↑",    CONF_MED),
                                     ("高信心", CONF_HIGH)]:
            r = backtest_threshold(df, line, conf_min)
            r["conf_filter"] = conf_label
            rows.append(r)
    return pd.DataFrame(rows)


# ── 報告輸出 ──────────────────────────────────────────────────

def print_summary(bt: pd.DataFrame):
    print(f"\n{'═'*65}")
    print(f"{'  Backtest Report — NBA Total Prediction':^65}")
    print(f"{'═'*65}")
    print(f"  {'線':>6}  {'篩選':<8}  {'場次':>5}  {'勝率':>7}  {'ROI':>7}  結果")
    print(f"  {'─'*59}")

    for _, row in bt.iterrows():
        if row["n_bets"] == 0 or row["win_rate"] is None:
            continue
        wr  = row["win_rate"]
        roi = row["roi"]

        if wr >= 0.55 and roi > 0:
            flag = "★ edge"
        elif wr >= BREAKEVEN_RATE:
            flag = "△ 平衡"
        else:
            flag = "✗"

        print(
            f"  {row['line']:>6.1f}  {row['conf_filter']:<8}  "
            f"{row['n_bets']:>5}  {wr:>6.1%}  {roi:>+6.1%}  {flag}"
        )
    print(f"{'═'*65}\n")


def print_season_breakdown(df: pd.DataFrame, line: float = 226.5):
    if "season" not in df.columns:
        log.warning("'season' column not in predictions, skipping season breakdown.")
        return
    print(f"\n  逐季分析（line={line}，高信心）：")
    print(f"  {'賽季':<10}  {'場次':>5}  {'勝率':>7}  {'ROI':>7}")
    print(f"  {'─'*38}")
    for season, sub in df.groupby("season"):
        r = backtest_threshold(sub, line, conf_min=CONF_HIGH)
        if r["n_bets"] == 0:
            continue
        print(f"  {season:<10}  {r['n_bets']:>5}  "
              f"{r['win_rate']:>6.1%}  {r['roi']:>+6.1%}")
    print()


def print_mae_by_range(df: pd.DataFrame):
    df = df.copy()
    df["error"] = (df["pred_ensemble"] - df["total_score"]).abs()
    df["bin"] = pd.cut(
        df["total_score"],
        bins=[180, 210, 220, 225, 230, 235, 240, 260],
        labels=["<210", "210-220", "220-225",
                "225-230", "230-235", "235-240", ">240"],
    )
    print("  MAE by total score range：")
    print(f"  {'區間':<12}  {'場次':>5}  {'MAE':>7}")
    print(f"  {'─'*28}")
    for label, group in df.groupby("bin", observed=True):
        print(f"  {str(label):<12}  {len(group):>5}  {group['error'].mean():>7.3f}")
    print()


# ── 儲存結果 ──────────────────────────────────────────────────

def save_results(bt: pd.DataFrame, df: pd.DataFrame):
    bt.to_csv(LOG_DIR / "backtest_results.csv", index=False)
    log.info(f"Saved → {LOG_DIR / 'backtest_results.csv'}")

    df_out = df.copy()
    for col in ["pred_ensemble", "pred_xgb", "pred_lgb", "pred_lstm"]:
        if col in df_out.columns:
            df_out[f"error_{col.replace('pred_','')}"] = \
                (df_out[col] - df_out["total_score"]).abs()
    df_out.to_csv(LOG_DIR / "predictions_detail.csv", index=False)
    log.info(f"Saved → {LOG_DIR / 'predictions_detail.csv'}")


# ── 進入點 ────────────────────────────────────────────────────

def main():
    df = load_predictions()
    bt = run_full_backtest(df)

    print_summary(bt)
    print_season_breakdown(df)
    print_mae_by_range(df)

    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(df["total_score"], df["pred_ensemble"])
    log.info(f"Overall Ensemble MAE: {mae:.3f}")

    save_results(bt, df)

    best = (bt[bt["n_bets"] >= 20]
            .dropna(subset=["win_rate"])
            .sort_values("win_rate", ascending=False)
            .head(5))
    if not best.empty:
        print("  Top 5 最佳組合（≥20場）：")
        print(best[["line", "conf_filter", "n_bets", "win_rate", "roi"]]
              .to_string(index=False))
        print()

    return bt


if __name__ == "__main__":
    main()
