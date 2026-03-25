"""
pregame_model.py
----------------
賽前模型：只使用比賽開始前可以取得的特徵。

與半場模型的差異：
  半場模型：用 h1_total（半場比分）→ MAE ~11，但需要等到半場
  賽前模型：不用任何盤中資料   → MAE 預計 ~14-16，但賽前即可預測

用途：
  搭配賽前賠率（The Odds API 免費方案）做有意義的回測

使用方式：
  python src/models/pregame_model.py
  python src/models/pregame_model.py --tune
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import pickle

import numpy as np
import pandas as pd
import shap
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)
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
SEED         = CFG["model"]["random_seed"]

# ── 賽前特徵（嚴格排除所有盤中資訊）────────────────────────────
# 這些欄位在比賽開始前完全不可知
INGAME_COLS = {
    # 各節比分
    "home_q1", "home_q2", "home_q3", "home_q4",
    "away_q1", "away_q2", "away_q3", "away_q4",
    # 最終得分
    "home_total", "away_total", "total_score",
    # 衍生的盤中特徵
    "h1_total", "q1_total", "q2_total",
    "h1_pace_ratio", "q1_pace_ratio",
    "q1_home_lead", "h1_home_lead",
    "q1_vs_avg",
}

# ID / 日期 / 非特徵欄位
META_COLS = {
    "game_id", "game_date", "season",
    "home_team", "away_team",
    "home_team_id", "away_team_id",
}

EXCLUDE_COLS = INGAME_COLS | META_COLS


def get_pregame_features(df: pd.DataFrame) -> list[str]:
    """回傳所有賽前可用的特徵欄位。"""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ── 資料準備 ──────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = FEAT_DIR / "features.csv"
    df = pd.read_csv(path, parse_dates=["game_date"])
    log.info(f"Loaded {len(df)} rows")
    return df


def split_data(df: pd.DataFrame):
    test_mask  = df["season"].isin(TEST_SEASONS)
    val_mask   = df["season"].isin(VAL_SEASONS)
    train_mask = ~test_mask & ~val_mask

    feat_cols = get_pregame_features(df)
    log.info(f"Pregame features: {len(feat_cols)}")

    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, "total_score"]
    X_val   = df.loc[val_mask,   feat_cols]
    y_val   = df.loc[val_mask,   "total_score"]
    X_test  = df.loc[test_mask,  feat_cols]
    y_test  = df.loc[test_mask,  "total_score"]

    log.info(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feat_cols


# ── XGBoost ───────────────────────────────────────────────────

def train_xgb(X_train, y_train, X_val, y_val,
              params: dict = None) -> xgb.XGBRegressor:
    p = params or {
        "n_estimators":     CFG["model"]["xgb"]["n_estimators"],
        "learning_rate":    CFG["model"]["xgb"]["learning_rate"],
        "max_depth":        CFG["model"]["xgb"]["max_depth"],
        "subsample":        CFG["model"]["xgb"]["subsample"],
        "colsample_bytree": CFG["model"]["xgb"]["colsample_bytree"],
        "random_state":     SEED,
        "n_jobs":           -1,
        "verbosity":        0,
    }
    model = xgb.XGBRegressor(**p)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    mae = mean_absolute_error(y_val, model.predict(X_val))
    log.info(f"XGB pregame val MAE: {mae:.3f}")
    return model


# ── LightGBM ──────────────────────────────────────────────────

def train_lgb(X_train, y_train, X_val, y_val,
              params: dict = None) -> lgb.LGBMRegressor:
    p = params or {
        "n_estimators":  CFG["model"]["lgb"]["n_estimators"],
        "learning_rate": CFG["model"]["lgb"]["learning_rate"],
        "num_leaves":    CFG["model"]["lgb"]["num_leaves"],
        "subsample":     CFG["model"]["lgb"]["subsample"],
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
    mae = mean_absolute_error(y_val, model.predict(X_val))
    log.info(f"LGB pregame val MAE: {mae:.3f}")
    return model


# ── Optuna 調參 ───────────────────────────────────────────────

def tune_lgb(X_train, y_train, X_val, y_val,
             n_trials: int = 50) -> dict:
    log.info(f"Tuning LGB pregame model ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0, 5),
            "random_state":     SEED,
            "n_jobs":           -1,
            "verbosity":        -1,
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])
        return mean_absolute_error(y_val, m.predict(X_val))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"Best val MAE: {study.best_value:.3f}")
    return study.best_params


# ── Ensemble（XGB + LGB 各 50%）──────────────────────────────

def ensemble_predict(xgb_model, lgb_model,
                     X: pd.DataFrame) -> np.ndarray:
    return 0.5 * xgb_model.predict(X) + 0.5 * lgb_model.predict(X)


# ── SHAP 分析 ─────────────────────────────────────────────────

def explain(model, X: pd.DataFrame, title: str = "", top_n: int = 15):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance  = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=False)

    print(f"\n{'─'*45}")
    print(f"  Top SHAP Features {title}")
    print(f"{'─'*45}")
    for feat, val in importance.head(top_n).items():
        bar = "█" * int(val / importance.iloc[0] * 20)
        print(f"  {feat:<38} {val:.3f}  {bar}")
    print(f"{'─'*45}\n")
    return importance


# ── 預測機率（用於回測）──────────────────────────────────────

def pred_to_prob_over(pred: float, line: float,
                      std: float = 13.0) -> float:
    """
    賽前模型 MAE 約 14-16，std 設 13（比半場模型寬）
    """
    return float(1 - norm.cdf(line, loc=pred, scale=std))


# ── 存讀模型 ──────────────────────────────────────────────────

def save(xgb_model, lgb_model, feat_cols: list):
    with open(MODEL_DIR / "pregame_xgb.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open(MODEL_DIR / "pregame_lgb.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    with open(MODEL_DIR / "pregame_meta.pkl", "wb") as f:
        pickle.dump({"feat_cols": feat_cols}, f)
    log.info(f"Pregame models saved → {MODEL_DIR}")


def load():
    with open(MODEL_DIR / "pregame_xgb.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(MODEL_DIR / "pregame_lgb.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    with open(MODEL_DIR / "pregame_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return xgb_model, lgb_model, meta["feat_cols"]


# ── 儲存測試集預測 ────────────────────────────────────────────

def save_test_predictions(df: pd.DataFrame,
                           xgb_model, lgb_model,
                           X_test: pd.DataFrame,
                           y_test: pd.Series):
    preds = ensemble_predict(xgb_model, lgb_model, X_test)
    test_df = df[df["season"].isin(TEST_SEASONS)].copy()
    test_df = test_df.reset_index(drop=True)
    result = test_df[["game_id", "game_date", "season",
                       "home_team", "away_team", "total_score"]].copy()
    result["pred_pregame"] = preds
    result["error"]        = (result["pred_pregame"] - result["total_score"]).abs()

    out = MODEL_DIR / "pregame_test_predictions.csv"
    result.to_csv(out, index=False)
    log.info(f"Saved → {out}")
    return result


# ── 進入點 ────────────────────────────────────────────────────

def main(do_tune: bool = False):
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = split_data(df)

    log.info("Training XGBoost (pregame)...")
    if do_tune:
        best = tune_lgb(X_train, y_train, X_val, y_val)
        lgb_model = train_lgb(X_train, y_train, X_val, y_val,
                              params={**best, "random_state": SEED,
                                      "n_jobs": -1, "verbosity": -1})
    else:
        lgb_model = train_lgb(X_train, y_train, X_val, y_val)

    xgb_model = train_xgb(X_train, y_train, X_val, y_val)

    # Ensemble 評估
    test_preds = ensemble_predict(xgb_model, lgb_model, X_test)
    test_mae   = mean_absolute_error(y_test, test_preds)

    log.info("SHAP analysis (LGB pregame)...")
    shap_imp = explain(lgb_model, X_val, title="(Pregame LGB)")

    # 與半場模型比較
    print(f"\n{'═'*45}")
    print(f"{'  Model Comparison':^45}")
    print(f"{'═'*45}")
    print(f"  {'Pregame Ensemble':<20} TEST MAE: {test_mae:.3f}")
    print(f"  {'Halftime LGB':<20} TEST MAE: 10.810  (reference)")
    print(f"  {'MAE difference':<20}          {test_mae - 10.810:+.3f}")
    print(f"\n  Pregame MAE 代表：賽前每場預測誤差 ±{test_mae:.1f} 分")
    print(f"{'═'*45}\n")

    save(xgb_model, lgb_model, feat_cols)
    result = save_test_predictions(df, xgb_model, lgb_model, X_test, y_test)
    log.info("Done. Next: python src/backtest/backtest_pregame.py")
    return xgb_model, lgb_model, feat_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    main(do_tune=args.tune)
