"""
baseline_model.py
-----------------
最簡單的 baseline：
  B1：第一節總分 × 4
  B2：半場總分 × 2
  B3：賽季平均（永遠猜平均值）

用來跟 ML 模型比較，確認模型有沒有真正學到東西。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import yaml

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

FEAT_DIR     = Path(__file__).parent.parent.parent / CFG["data"]["features_dir"]
TEST_SEASONS = CFG["model"]["test_seasons"]


def run():
    df = pd.read_csv(FEAT_DIR / "features.csv", parse_dates=["game_date"])
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    train = df[~df["season"].isin(TEST_SEASONS)].copy()

    actual = test["total_score"]

    # Baseline 1：Q1 × 4
    b1 = test["q1_total"] * 4
    mae_b1 = mean_absolute_error(actual, b1)

    # Baseline 2：H1 × 2
    b2 = test["h1_total"] * 2
    mae_b2 = mean_absolute_error(actual, b2)

    # Baseline 3：訓練集平均（賽前最簡單的猜法）
    train_mean = train["total_score"].mean()
    b3 = pd.Series(train_mean, index=test.index)
    mae_b3 = mean_absolute_error(actual, b3)

    # Baseline 4：上季平均（更貼近賽前的猜法）
    last_season_mean = df[df["season"] == "2022-23"]["total_score"].mean()
    b4 = pd.Series(last_season_mean, index=test.index)
    mae_b4 = mean_absolute_error(actual, b4)

    print(f"\n{'═'*52}")
    print(f"{'  Baseline Comparison':^52}")
    print(f"{'═'*52}")
    print(f"  {'模型':<30}  {'MAE':>7}  {'vs 賽前模型'}")
    print(f"  {'─'*46}")

    models = [
        ("B1: Q1 × 4（需要第一節）",    mae_b1),
        ("B2: H1 × 2（需要半場）",       mae_b2),
        ("B3: 訓練集平均（賽前猜法）",    mae_b3),
        ("B4: 上季平均（賽前猜法）",      mae_b4),
        ("ML 賽前模型（XGB+LGB）",       14.713),
        ("ML 半場模型（XGB+LGB）",       10.810),
    ]

    for name, mae in models:
        diff = mae - 14.713
        flag = f"  基準" if "賽前模型" in name else (
               f"  {diff:+.2f}" if "ML" not in name else "  ← 目標"
        )
        print(f"  {name:<30}  {mae:>7.3f}{flag}")

    print(f"{'═'*52}")
    print(f"\n  結論：")
    if mae_b1 > 14.713:
        print(f"  ✅ 賽前 ML 模型（14.71）比 Q1×4（{mae_b1:.2f}）更準")
    else:
        print(f"  ⚠️  Q1×4（{mae_b1:.2f}）比 ML 賽前模型還準，需要改善")

    if mae_b2 > 10.810:
        print(f"  ✅ 半場 ML 模型（10.81）比 H1×2（{mae_b2:.2f}）更準")
    else:
        print(f"  ⚠️  H1×2（{mae_b2:.2f}）比 ML 半場模型還準，需要改善")

    print()
    return {
        "q1x4": mae_b1,
        "h1x2": mae_b2,
        "train_mean": mae_b3,
        "last_season_mean": mae_b4,
    }


if __name__ == "__main__":
    run()
