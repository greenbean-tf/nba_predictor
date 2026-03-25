"""
xgb_model.py
------------
XGBoost 迴歸模型，預測比賽總得分。
支援 Optuna 自動調參、SHAP 特徵分析、模型存讀。

使用方式：
  python src/models/xgb_model.py
  python src/models/xgb_model.py --tune   # 開啟 Optuna 調參
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
import shap
import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)
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

TEST_SEASONS = CFG["model"]["test_seasons"]
VAL_SEASONS  = CFG["model"]["val_seasons"]
SEED         = CFG["model"]["random_seed"]
XGB_CFG      = CFG["model"]["xgb"]

# 特徵欄位（排除 ID、日期、目標）
EXCLUDE_COLS = {
    "game_id", "game_date", "season",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "total_score",                    # ← 預測目標
    "home_total", "away_total",       # ← target leakage
    "home_q3", "home_q4",            # ← 半場後不應使用 Q3/Q4
    "away_q3", "away_q4",
}


# ── 資料準備 ──────────────────────────────────────────────────

def load_data(use_full: bool = True) -> pd.DataFrame:
    fname = "features_full.csv" if use_full else "features.csv"
    path  = FEAT_DIR / fname
    if not path.exists():
        log.warning(f"{fname} not found, falling back to features.csv")
        path = FEAT_DIR / "features.csv"
    df = pd.read_csv(path, parse_dates=["game_date"])
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def split_data(df: pd.DataFrame) -> tuple:
    """
    時序切割（不可隨機 shuffle，避免 data leakage）：
      train : 非 val / test 賽季
      val   : VAL_SEASONS
      test  : TEST_SEASONS（最終評估，訓練中不可碰）
    """
    test_mask = df["season"].isin(TEST_SEASONS)
    val_mask  = df["season"].isin(VAL_SEASONS)
    train_mask = ~test_mask & ~val_mask

    feat_cols = get_feature_cols(df)
    target    = "total_score"

    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, target]
    X_val   = df.loc[val_mask,   feat_cols]
    y_val   = df.loc[val_mask,   target]
    X_test  = df.loc[test_mask,  feat_cols]
    y_test  = df.loc[test_mask,  target]

    log.info(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── 訓練 ──────────────────────────────────────────────────────

def train(X_train, y_train, X_val, y_val,
          params: dict = None) -> xgb.XGBRegressor:

    p = params or {
        "n_estimators":     XGB_CFG["n_estimators"],
        "learning_rate":    XGB_CFG["learning_rate"],
        "max_depth":        XGB_CFG["max_depth"],
        "subsample":        XGB_CFG["subsample"],
        "colsample_bytree": XGB_CFG["colsample_bytree"],
        "random_state":     SEED,
        "n_jobs":           -1,
        "verbosity":        0,
    }

    model = xgb.XGBRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_mae = mean_absolute_error(y_val, model.predict(X_val))
    log.info(f"XGBoost val MAE: {val_mae:.3f}")
    return model


# ── Optuna 自動調參 ───────────────────────────────────────────

def tune(X_train, y_train, X_val, y_val,
         n_trials: int = 50) -> dict:
    """用 Optuna 搜尋最佳超參數。"""
    log.info(f"Starting Optuna tuning ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "random_state":     SEED,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return mean_absolute_error(y_val, model.predict(X_val))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f"Best val MAE: {study.best_value:.3f}")
    log.info(f"Best params: {study.best_params}")
    return study.best_params


# ── SHAP 分析 ─────────────────────────────────────────────────

def explain(model: xgb.XGBRegressor,
            X: pd.DataFrame, top_n: int = 20):
    """印出 top N 最重要特徵（SHAP mean abs value）。"""
    log.info("Running SHAP analysis...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance  = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=False)

    print(f"\n{'─'*40}")
    print(f"{'Top SHAP Features':^40}")
    print(f"{'─'*40}")
    for feat, val in importance.head(top_n).items():
        bar = "█" * int(val / importance.iloc[0] * 20)
        print(f"  {feat:<35} {val:.3f}  {bar}")
    print(f"{'─'*40}\n")
    return importance


# ── 存讀模型 ──────────────────────────────────────────────────

def save(model: xgb.XGBRegressor, name: str = "xgb"):
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved → {path}")


def load(name: str = "xgb") -> xgb.XGBRegressor:
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ── 進入點 ────────────────────────────────────────────────────

def main(do_tune: bool = False):
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    if do_tune:
        best_params = tune(X_train, y_train, X_val, y_val)
        model = train(X_train, y_train, X_val, y_val, params={
            **best_params, "random_state": SEED, "n_jobs": -1, "verbosity": 0
        })
    else:
        model = train(X_train, y_train, X_val, y_val)

    # 最終測試集評估
    test_preds = model.predict(X_test)
    test_mae   = mean_absolute_error(y_test, test_preds)
    log.info(f"XGBoost TEST MAE: {test_mae:.3f}")

    # SHAP 分析
    explain(model, X_val)

    save(model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning")
    args = parser.parse_args()
    main(do_tune=args.tune)
