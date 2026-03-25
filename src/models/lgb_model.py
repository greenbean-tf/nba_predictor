"""
lgb_model.py
------------
LightGBM 迴歸模型，結構與 xgb_model.py 一致，
作為 Ensemble 的第二支腳。

使用方式：
  python src/models/lgb_model.py
  python src/models/lgb_model.py --tune
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
import lightgbm as lgb
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

FEAT_DIR  = Path(__file__).parent.parent.parent / CFG["data"]["features_dir"]
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEST_SEASONS = CFG["model"]["test_seasons"]
VAL_SEASONS  = CFG["model"]["val_seasons"]
SEED         = CFG["model"]["random_seed"]
LGB_CFG      = CFG["model"]["lgb"]

EXCLUDE_COLS = {
    "game_id", "game_date", "season",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "total_score",
    "home_total", "away_total",
    "home_q3", "home_q4",
    "away_q3", "away_q4",
}


# ── 資料準備 ──────────────────────────────────────────────────

def load_data(use_full: bool = True) -> pd.DataFrame:
    fname = "features_full.csv" if use_full else "features.csv"
    path  = FEAT_DIR / fname
    if not path.exists():
        path = FEAT_DIR / "features.csv"
    df = pd.read_csv(path, parse_dates=["game_date"])
    log.info(f"Loaded {len(df)} rows from {path.name}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def split_data(df: pd.DataFrame) -> tuple:
    test_mask  = df["season"].isin(TEST_SEASONS)
    val_mask   = df["season"].isin(VAL_SEASONS)
    train_mask = ~test_mask & ~val_mask
    feat_cols  = get_feature_cols(df)
    target     = "total_score"

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
          params: dict = None) -> lgb.LGBMRegressor:

    p = params or {
        "n_estimators":  LGB_CFG["n_estimators"],
        "learning_rate": LGB_CFG["learning_rate"],
        "num_leaves":    LGB_CFG["num_leaves"],
        "subsample":     LGB_CFG["subsample"],
        "random_state":  SEED,
        "n_jobs":        -1,
        "verbosity":     -1,
    }

    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    val_mae = mean_absolute_error(y_val, model.predict(X_val))
    log.info(f"LightGBM val MAE: {val_mae:.3f}")
    return model


# ── Optuna 自動調參 ───────────────────────────────────────────

def tune(X_train, y_train, X_val, y_val,
         n_trials: int = 50) -> dict:
    log.info(f"Starting Optuna tuning ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":       trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":      trial.suggest_float("reg_lambda", 0, 5),
            "random_state":    SEED,
            "n_jobs":          -1,
            "verbosity":       -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        return mean_absolute_error(y_val, model.predict(X_val))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f"Best val MAE: {study.best_value:.3f}")
    log.info(f"Best params: {study.best_params}")
    return study.best_params


# ── SHAP 分析 ─────────────────────────────────────────────────

def explain(model: lgb.LGBMRegressor,
            X: pd.DataFrame, top_n: int = 20):
    log.info("Running SHAP analysis...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance  = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=False)

    print(f"\n{'─'*40}")
    print(f"{'Top SHAP Features (LGB)':^40}")
    print(f"{'─'*40}")
    for feat, val in importance.head(top_n).items():
        bar = "█" * int(val / importance.iloc[0] * 20)
        print(f"  {feat:<35} {val:.3f}  {bar}")
    print(f"{'─'*40}\n")
    return importance


# ── 存讀模型 ──────────────────────────────────────────────────

def save(model: lgb.LGBMRegressor, name: str = "lgb"):
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved → {path}")


def load(name: str = "lgb") -> lgb.LGBMRegressor:
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
            **best_params, "random_state": SEED, "n_jobs": -1, "verbosity": -1
        })
    else:
        model = train(X_train, y_train, X_val, y_val)

    test_preds = model.predict(X_test)
    test_mae   = mean_absolute_error(y_test, test_preds)
    log.info(f"LightGBM TEST MAE: {test_mae:.3f}")

    explain(model, X_val)
    save(model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    main(do_tune=args.tune)
