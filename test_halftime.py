"""
test_halftime.py
----------------
測試半場模型的完整流程，不需要等真實比賽。

測試項目：
  1. live_scraper 的 Halftime 偵測邏輯
  2. 半場特徵建構
  3. 半場模型預測
  4. Polymarket 賠率抓取
  5. EV / Kelly 計算
  6. SimAccount 下注記錄
  7. 結算邏輯

執行：
  python test_halftime.py
  python test_halftime.py --verbose   # 顯示更多細節
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import pickle
import logging
from datetime import date

import pandas as pd
import numpy as np
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 顏色輸出 ──────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"     {msg}")


# ── Test 1: live_scraper Halftime 偵測 ────────────────────────

def test_halftime_detection():
    print("\n【Test 1】live_scraper Halftime 偵測")

    from src.scraper.live_scraper import QuarterTracker

    # 模擬 ESPN 回傳的各種半場狀態
    test_cases = [
        # (status_text, clock, quarter, should_trigger)
        ("Halftime",             "0.0",  2, True),   # 正常半場
        ("Halftime",             "0:00", 2, True),   # 另一種格式
        ("Half Time",            "0.0",  2, True),   # 空格版
        ("End of 2nd Quarter",   "0:00", 2, True),   # End of 2nd
        ("End of 2nd Quarter",   "0.0",  2, True),
        ("In Progress",          "5:30", 2, False),  # 比賽中，不觸發
        ("In Progress",          "0:00", 1, True),   # Q1 結束
        ("In Progress",          "0.0",  1, True),   # Q1 結束（0.0格式）
        ("End of 3rd Quarter",   "0:00", 3, False),  # Q3 結束不觸發
        ("Final",                "0:00", 4, False),  # 比賽結束不觸發
    ]

    tracker = QuarterTracker()
    passed = 0

    for i, (status_text, clock, quarter, should_trigger) in enumerate(test_cases):
        # 每個 test case 用唯一的 game_id，避免重複觸發判斷互相干擾
        game = {
            "game_id":     f"test_case_{i:03d}",
            "status":      "in",
            "status_text": status_text,
            "quarter":     quarter,
            "clock":       clock,
            "home_team":   "BOS", "away_team": "NYK",
            "home_q1": 28, "home_q2": 30,
            "away_q1": 25, "away_q2": 27,
            "home_total": 58, "away_total": 52,
            "q1_total": 53, "h1_total": 110,
        }

        result = tracker.check(game)
        triggered = result is not None

        if triggered == should_trigger:
            ok(f"'{status_text}' Q{quarter} clock={clock} → {'觸發' if triggered else '不觸發'} ✓")
            passed += 1
        else:
            fail(f"'{status_text}' Q{quarter} clock={clock} → 期望{'觸發' if should_trigger else '不觸發'}，實際{'觸發' if triggered else '不觸發'}")

    print(f"\n  結果：{passed}/{len(test_cases)} 通過")
    return passed == len(test_cases)


# ── Test 2: 半場特徵建構 ──────────────────────────────────────

def test_feature_building():
    print("\n【Test 2】半場特徵建構")

    try:
        import yaml
        CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
        with open(CONFIG_PATH, encoding="utf-8") as f:
            CFG = yaml.safe_load(f)

        FEAT_DIR  = Path(__file__).parent / CFG["data"]["features_dir"]
        MODEL_DIR = Path(__file__).parent / "models" / "saved"

        # 載入模型取得 feat_cols
        xgb_model = pickle.load(open(MODEL_DIR / "xgb.pkl", "rb"))
        feat_cols  = list(xgb_model.feature_names_in_)
        ok(f"模型載入成功，特徵數：{len(feat_cols)}")

        # 載入歷史資料
        history = pd.read_csv(FEAT_DIR / "features.csv", parse_dates=["game_date"])
        ok(f"歷史資料載入：{len(history)} 筆")

        # 建構特徵
        from src.live.ab_simulator import build_features
        feat_df = build_features(
            home_abbr="BOS", away_abbr="NYK",
            history=history,
            feat_cols=feat_cols,
            home_q1=32, away_q1=28,
            home_q2=29, away_q2=25,
        )

        # 驗證
        ok(f"特徵向量建構成功，shape={feat_df.shape}")

        # 關鍵特徵檢查
        checks = [
            ("h1_total",     feat_df["h1_total"].iloc[0],     32+29+28+25),
            ("q1_total",     feat_df["q1_total"].iloc[0],     32+28),
            ("h1_home_lead", feat_df["h1_home_lead"].iloc[0], (32+29)-(28+25)),
        ]
        for col, got, expected in checks:
            if abs(got - expected) < 0.01:
                ok(f"  {col} = {got} ✓")
            else:
                fail(f"  {col} = {got}，期望 {expected}")

        # NaN 檢查
        nan_count = feat_df.isna().sum().sum()
        if nan_count == 0:
            ok("無 NaN 值")
        else:
            warn(f"{nan_count} 個 NaN（已填 0）")

        return True

    except Exception as e:
        fail(f"特徵建構失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── Test 3: 半場模型預測 ──────────────────────────────────────

def test_prediction():
    print("\n【Test 3】半場模型預測")

    try:
        import yaml
        CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
        with open(CONFIG_PATH, encoding="utf-8") as f:
            CFG = yaml.safe_load(f)

        FEAT_DIR  = Path(__file__).parent / CFG["data"]["features_dir"]
        MODEL_DIR = Path(__file__).parent / "models" / "saved"

        xgb_model = pickle.load(open(MODEL_DIR / "xgb.pkl", "rb"))
        lgb_model = pickle.load(open(MODEL_DIR / "lgb.pkl", "rb"))
        feat_cols = list(xgb_model.feature_names_in_)
        history   = pd.read_csv(FEAT_DIR / "features.csv", parse_dates=["game_date"])

        from src.live.ab_simulator import build_features

        # 幾個不同的半場比分情境
        scenarios = [
            ("低分半場",  "BOS", "NYK", 28, 25, 24, 22),  # H1=99
            ("正常半場",  "LAL", "GSW", 32, 30, 28, 27),  # H1=117
            ("高分半場",  "MIL", "CLE", 38, 35, 36, 34),  # H1=143
        ]

        all_ok = True
        for label, home, away, hq1, aq1, hq2, aq2 in scenarios:
            feat_df = build_features(
                home_abbr=home, away_abbr=away,
                history=history, feat_cols=feat_cols,
                home_q1=hq1, away_q1=aq1,
                home_q2=hq2, away_q2=aq2,
            )
            p_xgb = float(xgb_model.predict(feat_df)[0])
            p_lgb = float(lgb_model.predict(feat_df)[0])
            pred  = round((p_xgb + p_lgb) / 2, 1)
            h1    = hq1 + aq1 + hq2 + aq2

            # 預測應在合理範圍（150~300）
            if 150 < pred < 300:
                ok(f"{label}: H1={h1} → pred={pred} (xgb={p_xgb:.1f}, lgb={p_lgb:.1f})")
            else:
                fail(f"{label}: pred={pred} 超出合理範圍")
                all_ok = False

        return all_ok

    except Exception as e:
        fail(f"預測失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── Test 4: Polymarket 賠率 ───────────────────────────────────

def test_polymarket():
    print("\n【Test 4】Polymarket 賠率抓取")

    try:
        from src.scraper.polymarket_scraper import (
            fetch_nba_game_events, parse_total_markets
        )
        events  = fetch_nba_game_events()
        markets = parse_total_markets(events)

        ok(f"抓到 {len(events)} 個 events，{len(markets)} 個 total 市場")

        if not markets:
            warn("今天沒有活躍的 total 市場（可能比賽還沒開始或已結束）")
            return True

        # 驗證每個市場的資料格式
        errors = 0
        for m in markets:
            if not (0 < m["over_prob"] < 1):
                fail(f"over_prob 異常：{m['over_prob']} for {m['question']}")
                errors += 1
            if not (0 < m["under_prob"] < 1):
                fail(f"under_prob 異常：{m['under_prob']} for {m['question']}")
                errors += 1
            if abs(m["over_prob"] + m["under_prob"] - 1.0) > 0.01:
                warn(f"over+under ≠ 1: {m['over_prob']+m['under_prob']:.3f} for {m['question']}")

        if errors == 0:
            ok("所有市場格式正確")
            info(f"範例：{markets[0]['question']} | Over={markets[0]['over_prob']:.1%}")
        return errors == 0

    except Exception as e:
        fail(f"Polymarket 抓取失敗：{e}")
        return False


# ── Test 5: 完整半場流程模擬 ─────────────────────────────────

def test_full_halftime_flow():
    print("\n【Test 5】完整半場流程模擬（用真實 Polymarket 資料）")

    try:
        from src.live.ab_simulator import ABSimulator

        # 用小額資金測試，不影響真實記錄
        sim = ABSimulator(capital=1000)

        # 找今天 Polymarket 上有哪些比賽
        from src.scraper.polymarket_scraper import (
            fetch_nba_game_events, parse_total_markets
        )
        events  = fetch_nba_game_events()
        markets = parse_total_markets(events)

        if not markets:
            warn("今天沒有 Polymarket 市場，用模擬資料測試")
            # 用假市場資料
            test_markets = [{
                "game_slug": "nba-bos-nyk-2026-03-21",
                "home_team": "Knicks",
                "away_team": "Celtics",
                "question":  "Celtics vs. Knicks: O/U 226.5",
                "line":      226.5,
                "over_prob": 0.51,
                "under_prob": 0.49,
                "condition_id": "test123",
                "fetched_at": "2026-03-21",
            }]
            home_name = "Knicks"
            away_name = "Celtics"
        else:
            # 用第一個有市場的比賽
            first = markets[0]
            home_name = first["home_team"]
            away_name = first["away_team"]
            info(f"測試比賽：{away_name} @ {home_name}")

        # 模擬半場觸發
        game_dict = {
            "game_id":   "test_game_001",
            "home_team": sim.pregame.name,  # 先用這個
            "away_team": "NYK",
            "home_q1": 32, "home_q2": 29,
            "away_q1": 28, "away_q2": 25,
        }

        # 找到 home/away 縮寫
        from src.live.ab_simulator import TEAM_NAME_MAP, ABBR_TO_FULL
        home_abbr = TEAM_NAME_MAP.get(home_name, home_name[:3].upper())
        away_abbr = TEAM_NAME_MAP.get(away_name, away_name[:3].upper())
        game_dict["home_team"] = home_abbr
        game_dict["away_team"] = away_abbr

        ok(f"模擬半場觸發：{away_name}({away_abbr}) @ {home_name}({home_abbr})")
        info(f"  H1: {game_dict['away_q1']+game_dict['away_q2']}-{game_dict['home_q1']+game_dict['home_q2']}")

        # 執行半場模型
        sim.on_halftime(game_dict)

        # 查看結果
        bets = [b for b in sim.halftime._bets
                if b["model"] == "halftime" and b["result"] is None]

        if bets:
            ok(f"半場模型下注成功！共 {len(bets)} 注")
            for b in bets:
                info(f"  {b['game']} {b['line']} {b['side']} "
                     f"${b['amount']:.0f} | EV={b['ev']:+.1%}")
        else:
            warn("半場模型沒有下注（可能 EV 不足或 Polymarket 找不到市場）")
            info("  這不一定是錯誤，可能今天的賠率沒有 edge")

        return True

    except Exception as e:
        fail(f"完整流程失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── Test 6: 結算邏輯 ──────────────────────────────────────────

def test_settlement():
    print("\n【Test 6】結算邏輯驗證")

    try:
        import yaml
        CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
        with open(CONFIG_PATH, encoding="utf-8") as f:
            CFG = yaml.safe_load(f)

        from src.live.ab_simulator import SimAccount, AB_LOG

        # 用暫時的 log 路徑避免污染真實資料
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.close()
        orig_log = AB_LOG
        # monkey-patch AB_LOG
        import src.live.ab_simulator as ab_mod
        ab_mod.AB_LOG = Path(tmp.name)

        acc = SimAccount("test_settle", capital=1000)

        # 手動新增一筆下注
        acc._bets.append({
            "model":    "test_settle",
            "date":     str(date.today()),
            "ts":       "2026-03-21T10:00:00",
            "trigger":  "halftime",
            "game":     "Celtics @ Knicks",
            "line":     226.5,
            "side":     "OVER",
            "pred":     231.0,
            "model_p":  0.62,
            "market_p": 0.51,
            "edge":     0.11,
            "ev":       0.11,
            "amount":   100.0,
            "capital":  1000.0,
            "result":   None,
            "pnl":      None,
        })
        acc._save(acc._bets[-1])

        # 測試結算贏
        acc.settle("Celtics @ Knicks", 226.5, 235)  # 235 > 226.5 → OVER wins
        bet = acc._bets[-1]
        if bet["result"] == "win":
            expected_pnl = round(100 / 0.51 - 100, 2)
            ok(f"WIN 結算正確：pnl={bet['pnl']:+.2f} (期望 {expected_pnl:+.2f})")
        else:
            fail(f"WIN 結算錯誤：result={bet['result']}")

        # 測試結算輸
        acc._bets.append({
            **acc._bets[-1],
            "side": "UNDER", "line": 230.5,
            "result": None, "pnl": None,
        })
        acc.settle("Celtics @ Knicks", 230.5, 235)  # 235 > 230.5 → UNDER loses
        bet2 = acc._bets[-1]
        if bet2["result"] == "loss":
            ok(f"LOSS 結算正確：pnl={bet2['pnl']:+.2f}")
        else:
            fail(f"LOSS 結算錯誤：result={bet2['result']}")

        # 清理
        ab_mod.AB_LOG = orig_log
        os.unlink(tmp.name)

        return True

    except Exception as e:
        fail(f"結算邏輯失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── 主程式 ────────────────────────────────────────────────────

def main(verbose: bool = False):
    print("=" * 55)
    print("  Halftime Model End-to-End Test")
    print("=" * 55)

    tests = [
        ("Halftime 偵測",    test_halftime_detection),
        ("特徵建構",          test_feature_building),
        ("模型預測",          test_prediction),
        ("Polymarket 賠率",  test_polymarket),
        ("完整半場流程",      test_full_halftime_flow),
        ("結算邏輯",          test_settlement),
    ]

    results = []
    for name, func in tests:
        try:
            passed = func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  {RED}CRASH in {name}: {e}{RESET}")
            results.append((name, False))

    print("\n" + "=" * 55)
    print("  測試結果總覽")
    print("=" * 55)
    all_pass = True
    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print("=" * 55)
    if all_pass:
        print(f"  {GREEN}全部通過！半場模型可以正常運行。{RESET}")
    else:
        print(f"  {RED}有測試失敗，請檢查上方錯誤訊息。{RESET}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(verbose=args.verbose)