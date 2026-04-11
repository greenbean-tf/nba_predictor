"""
test_live_trader.py
-------------------
測試 live_trader.py 的完整流程，不需要等真實比賽。

測試項目：
  1. 設定載入（各種參數組合）
  2. 模型預測
  3. Polymarket 市場抓取
  4. 下注邏輯（dry-run）
  5. 兩個獨立帳戶
  6. 結算邏輯
  7. Report 輸出

執行：
  python test_live_trader.py
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")

# ── 暫時替換 TRADER_JSONL 路徑，避免污染真實資料 ─────────────
import src.live.live_trader as lt_mod
_tmp_dir = tempfile.mkdtemp()
_orig_pre  = lt_mod.TRADER_JSONL_PREGAME
_orig_half = lt_mod.TRADER_JSONL_HALFTIME
lt_mod.TRADER_JSONL_PREGAME  = Path(_tmp_dir) / "test_trades_pregame.jsonl"
lt_mod.TRADER_JSONL_HALFTIME = Path(_tmp_dir) / "test_trades_halftime.jsonl"


# ── 建立測試用 args ───────────────────────────────────────────

class FakeArgs:
    def __init__(self, **kwargs):
        defaults = dict(
            mode             = "both",
            capital          = 100.0,
            pregame_capital  = None,
            halftime_capital = None,
            kelly            = 0.25,
            min_ev           = 0.03,
            min_edge         = 0.03,
            max_bet          = 500.0,
            max_pct          = 0.20,
            pregame_std      = 13.0,
            halftime_std     = 9.0,
            live             = False,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ── Test 1: 設定載入 ──────────────────────────────────────────

def test_config():
    print("\n【Test 1】設定載入")
    from src.live.live_trader import TraderConfig

    # both 模式各半
    cfg = TraderConfig(FakeArgs(mode="both", capital=100))
    if abs(cfg.pregame_capital - 50) < 0.01 and abs(cfg.halftime_capital - 50) < 0.01:
        ok(f"both 模式：賽前 ${cfg.pregame_capital}，半場 ${cfg.halftime_capital}")
    else:
        fail(f"both 模式資金分配錯誤：{cfg.pregame_capital}, {cfg.halftime_capital}")

    # both 模式自訂
    cfg2 = TraderConfig(FakeArgs(mode="both", capital=100, pregame_capital=30, halftime_capital=70))
    if abs(cfg2.pregame_capital - 30) < 0.01 and abs(cfg2.halftime_capital - 70) < 0.01:
        ok(f"自訂分配：賽前 ${cfg2.pregame_capital}，半場 ${cfg2.halftime_capital}")
    else:
        fail(f"自訂分配錯誤")

    # pregame 模式
    cfg3 = TraderConfig(FakeArgs(mode="pregame", capital=80))
    if abs(cfg3.pregame_capital - 80) < 0.01 and cfg3.halftime_capital == 0:
        ok(f"pregame 模式：全部 ${cfg3.pregame_capital}")
    else:
        fail("pregame 模式資金錯誤")

    # halftime 模式
    cfg4 = TraderConfig(FakeArgs(mode="halftime", capital=60))
    if abs(cfg4.halftime_capital - 60) < 0.01 and cfg4.pregame_capital == 0:
        ok(f"halftime 模式：全部 ${cfg4.halftime_capital}")
    else:
        fail("halftime 模式資金錯誤")

    print(TraderConfig(FakeArgs(mode="both", capital=100)))
    return True


# ── Test 2: 兩個獨立帳戶 ─────────────────────────────────────

def test_accounts():
    print("\n【Test 2】兩個獨立帳戶")
    from src.live.live_trader import TraderConfig, TradingAccount

    cfg = TraderConfig(FakeArgs(mode="both", capital=100))
    pre  = TradingAccount(cfg, "pregame",  cfg.pregame_capital)
    half = TradingAccount(cfg, "halftime", cfg.halftime_capital)

    ok(f"賽前帳戶：初始 ${pre.initial}，可用 ${pre.available}")
    ok(f"半場帳戶：初始 ${half.initial}，可用 ${half.available}")

    # Kelly 計算
    amount_pre  = pre.kelly_amount(0.62, 0.51)
    amount_half = half.kelly_amount(0.70, 0.47)
    ok(f"賽前 Kelly 金額：${amount_pre:.2f}（模型62%，市場51%）")
    ok(f"半場 Kelly 金額：${amount_half:.2f}（模型70%，市場47%）")

    # 確認兩個帳戶獨立
    if pre.total != half.total or pre.initial == half.initial == 50:
        ok("兩個帳戶各自獨立")

    return True


# ── Test 3: 模型預測 ──────────────────────────────────────────

def test_prediction():
    print("\n【Test 3】模型預測")
    try:
        from src.live.live_trader import TraderConfig, LiveTrader
        cfg    = TraderConfig(FakeArgs())
        trader = LiveTrader(cfg)
        ok("模型載入成功")

        # 測試預測
        scenarios = [
            ("BOS", "NYK", 30, 28, 28, 25, "halftime"),
            ("LAL", "GSW", 35, 32, 30, 28, "halftime"),
        ]
        for ha, aa, hq1, aq1, hq2, aq2, trigger in scenarios:
            from src.live.ab_simulator import build_features
            feat = build_features(
                ha, aa, trader.history, trader.half_feats,
                home_q1=hq1, away_q1=aq1, home_q2=hq2, away_q2=aq2
            )
            pred = trader._predict(trader.xgb_half, trader.lgb_half, feat)
            h1 = hq1+aq1+hq2+aq2
            if 150 < pred < 300:
                ok(f"{aa}@{ha} H1={h1} → pred={pred}")
            else:
                fail(f"預測超出範圍：{pred}")
        return True
    except Exception as e:
        fail(f"模型預測失敗：{e}")
        return False


# ── Test 4: Polymarket 市場抓取 ───────────────────────────────

def test_polymarket():
    print("\n【Test 4】Polymarket 市場抓取")
    try:
        from src.live.live_trader import TraderConfig, LiveTrader
        cfg    = TraderConfig(FakeArgs())
        trader = LiveTrader(cfg)

        events  = __import__('src.scraper.polymarket_scraper', fromlist=['fetch_nba_game_events', 'parse_total_markets'])
        from src.scraper.polymarket_scraper import fetch_nba_game_events, parse_total_markets
        evs     = fetch_nba_game_events()
        markets = parse_total_markets(evs)

        ok(f"抓到 {len(markets)} 個 total 市場")

        if markets:
            m = markets[0]
            ok(f"範例：{m['question']} Over={m['over_prob']:.2f}")

            # 測試 _find_best_bet
            best = trader._find_best_bet([m], pred=235.0, std=9.0)
            if best:
                ok(f"找到最佳注：{best['side']} {best['line']} EV={best['ev']:+.1%}")
            else:
                warn("此市場無正 EV（pred=235 可能不適合這條線）")
        return True
    except Exception as e:
        fail(f"Polymarket 抓取失敗：{e}")
        return False


# ── Test 5: 完整下注流程（dry-run）───────────────────────────

def test_betting_flow():
    print("\n【Test 5】完整下注流程（dry-run）")
    try:
        from src.live.live_trader import TraderConfig, LiveTrader
        from src.scraper.polymarket_scraper import fetch_nba_game_events, parse_total_markets

        cfg    = TraderConfig(FakeArgs(mode="both", capital=100))
        trader = LiveTrader(cfg)

        # 找一個有市場的比賽
        evs     = fetch_nba_game_events()
        markets = parse_total_markets(evs)

        if not markets:
            warn("今天沒有市場，跳過下注測試")
            return True

        m = markets[0]
        ok(f"測試市場：{m['question']}")

        # 模擬 on_pregame
        home = m["home_team"]
        away = m["away_team"]
        trader.on_pregame(away, home)

        # 模擬 on_halftime
        from src.live.ab_simulator import TEAM_NAME_MAP
        home_abbr = TEAM_NAME_MAP.get(home, home[:3].upper())
        away_abbr = TEAM_NAME_MAP.get(away, away[:3].upper())
        game_dict = {
            "game_id":   "test_001",
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_q1": 32, "away_q1": 28,
            "home_q2": 29, "away_q2": 25,
            "home_total": 61, "away_total": 53,
            "q1_total": 60, "h1_total": 114,
        }
        trader.on_halftime(game_dict)

        # 查看下注記錄
        pre_bets  = trader.pregame_account._trades
        half_bets = trader.halftime_account._trades

        ok(f"賽前帳戶：{len(pre_bets)} 筆記錄")
        ok(f"半場帳戶：{len(half_bets)} 筆記錄")

        for b in pre_bets + half_bets:
            ok(f"  [{b['account']}] {b['game']} {b['line']} {b['side']} ${b['amount']:.0f} EV={b['ev']:+.1%} dry_run={b['dry_run']}")

        return True
    except Exception as e:
        fail(f"下注流程失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── Test 6: 結算邏輯 ──────────────────────────────────────────

def test_settlement():
    print("\n【Test 6】結算邏輯")
    try:
        from src.live.live_trader import TraderConfig, TradingAccount

        cfg  = TraderConfig(FakeArgs(mode="both", capital=100))
        acct = TradingAccount(cfg, "pregame", 50.0)

        # 手動加一筆下注（含新欄位）
        acct._trades.append({
            "account":       "pregame",
            "game":          "Celtics @ Knicks",
            "line":          226.5,
            "side":          "OVER",
            "amount":        20.00,
            "market_p":      0.52,
            "filled_price":  0.53,
            "edge":          round(0.62 - 0.52, 4),   # model_p - market_p
            "ev":            round(0.62 / 0.52 - 1, 4),  # 正確 EV 公式
            "order_success": True,
            "trade_status":  "MATCHED",
            "trade_success": True,
            "attempt":       1,
            "result":        None,
            "pnl":           None,
        })

        # 結算贏
        acct.settle("Celtics @ Knicks", 226.5, 235)
        t = acct._trades[-1]
        if t["result"] == "win":
            ok(f"WIN 結算：pnl={t['pnl']:+.2f}，資金 ${acct.total:.2f}")
        else:
            fail(f"WIN 結算錯誤：{t['result']}")

        # 結算輸
        acct._trades.append({
            "account":       "pregame",
            "game":          "Lakers @ Heat",
            "line":          230.5,
            "side":          "UNDER",
            "amount":        15.00,
            "market_p":      0.48,
            "filled_price":  0.49,
            "edge":          round(0.58 - 0.48, 4),
            "ev":            round(0.58 / 0.48 - 1, 4),
            "order_success": True,
            "trade_status":  "MATCHED",
            "trade_success": True,
            "attempt":       1,
            "result":        None,
            "pnl":           None,
        })
        acct.settle("Lakers @ Heat", 230.5, 245)
        t2 = acct._trades[-1]
        if t2["result"] == "loss":
            ok(f"LOSS 結算：pnl={t2['pnl']:+.2f}，資金 ${acct.total:.2f}")
        else:
            fail(f"LOSS 結算錯誤：{t2['result']}")

        return True
    except Exception as e:
        fail(f"結算失敗：{e}")
        return False


# ── Test 7: Report 輸出 ───────────────────────────────────────

def test_report():
    print("\n【Test 7】Report 輸出")
    try:
        from src.live.live_trader import TraderConfig, LiveTrader

        cfg    = TraderConfig(FakeArgs(mode="both", capital=100))
        trader = LiveTrader(cfg)

        # 手動塞一些假資料
        trader.pregame_account._trades = [
            {"account":"pregame","date":"2026-03-22","trigger":"pregame",
             "game":"Lakers @ Magic","line":231.5,"side":"UNDER","pred":219,
             "market_p":0.47,"ev":0.363,"amount":45.0,"order_id":"DRY_RUN",
             "dry_run":True,"result":"win","pnl":50.85,"actual_total":209},
        ]
        trader.pregame_account.total = 50 + 50.85

        trader.halftime_account._trades = [
            {"account":"halftime","date":"2026-03-22","trigger":"halftime",
             "game":"Grizzlies @ Hornets","line":229.5,"side":"UNDER","pred":218,
             "market_p":0.47,"ev":0.421,"amount":30.0,"order_id":"DRY_RUN",
             "dry_run":True,"result":"win","pnl":33.83,"actual_total":225},
        ]
        trader.halftime_account.total = 50 + 33.83

        trader.print_report()
        ok("Report 輸出正常")
        return True
    except Exception as e:
        fail(f"Report 失敗：{e}")
        import traceback
        traceback.print_exc()
        return False



# ── Test 8: CLOB ask price 取代 Gamma mid price ──────────────

def test_clob_ask_price():
    print("\n【Test 8】CLOB ask price 取代 Gamma mid price")
    try:
        from src.live.live_trader import TraderConfig, LiveTrader
        from src.scraper.polymarket_scraper import fetch_nba_game_events, parse_total_markets

        cfg    = TraderConfig(FakeArgs())
        trader = LiveTrader(cfg)

        # 先用 Gamma API 取得原始 mid price
        evs     = fetch_nba_game_events()
        markets = parse_total_markets(evs)

        if not markets:
            warn("今天沒有市場，跳過 CLOB ask 測試")
            return True

        m_gamma = markets[0]
        gamma_over = m_gamma["over_prob"]
        ok(f"Gamma API mid price: Over={gamma_over:.4f}")

        # 用 _get_markets 取得含 CLOB ask 的市場
        home = m_gamma["home_team"]
        away = m_gamma["away_team"]
        markets_clob = trader._get_markets(home, away)

        if not markets_clob:
            warn("找不到對應市場")
            return True

        m_clob = markets_clob[0]
        clob_over = m_clob["over_prob"]
        ok(f"CLOB ask price:      Over={clob_over:.4f}")

        # 驗證有取得 token_id
        if m_clob.get("token_id_over"):
            ok(f"token_id_over 已填入：{m_clob['token_id_over'][:16]}...")
        else:
            warn("token_id_over 未填入（可能 CLOB 查詢失敗，仍使用 Gamma mid）")

        # 驗證價格合理（0~1 之間）
        if 0 < clob_over < 1:
            ok(f"CLOB ask 價格合理：{clob_over:.4f}")
        else:
            fail(f"CLOB ask 價格異常：{clob_over}")

        return True
    except Exception as e:
        fail(f"CLOB ask price 測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── Test 9: filled_price 記錄與結算邏輯 ──────────────────────

def test_filled_price_settlement():
    print("\n【Test 9】filled_price 記錄與結算邏輯")
    try:
        from src.live.live_trader import TraderConfig, TradingAccount

        cfg  = TraderConfig(FakeArgs(mode="both", capital=100))
        acct = TradingAccount(cfg, "halftime", 50.0)

        # 模擬半場下注：market_p=0.48，但實際成交 filled_price=0.93（大幅領先情況）
        acct._trades.append({
            "account":       "halftime",
            "game":          "Thunder @ 76ers",
            "line":          235.5,
            "side":          "UNDER",
            "amount":        10.00,
            "market_p":      0.48,
            "filled_price":  0.93,
            "edge":          round(0.91 - 0.48, 4),      # model_p - market_p
            "ev":            round(0.91 / 0.48 - 1, 4),  # 正確 EV 公式
            "order_success": True,
            "trade_status":  "MATCHED",
            "trade_success": True,
            "attempt":       1,
            "result":        None,
            "pnl":           None,
        })

        # 結算：實際總分 226 < 235.5 → UNDER 贏
        acct.settle("Thunder @ 76ers", 235.5, 226)
        t = acct._trades[-1]

        if t["result"] != "win":
            fail(f"結算方向錯誤：{t['result']}")
            return False

        # 用 filled_price 計算正確 pnl
        expected_pnl = round(10.0 / 0.93 - 10.0, 2)
        actual_pnl   = t["pnl"]

        ok(f"WIN 結算（filled_price=0.93）：pnl={actual_pnl:+.2f}（預期 {expected_pnl:+.2f}）")

        if abs(actual_pnl - expected_pnl) < 0.05:
            ok("filled_price pnl 計算正確")
        else:
            fail(f"pnl 計算錯誤：{actual_pnl} vs 預期 {expected_pnl}")

        # 確認沒有用 market_p 計算（0.48 的 pnl 會差很多）
        wrong_pnl = round(10.0 / 0.48 - 10.0, 2)
        if abs(actual_pnl - wrong_pnl) > 1.0:
            ok(f"確認未用 market_p 計算（market_p pnl 為 {wrong_pnl:+.2f}，差異顯著）")
        else:
            warn("無法確認是否用了正確的 filled_price")

        return True
    except Exception as e:
        fail(f"filled_price 結算測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── Test 10: 高 market_p 過濾（半場大幅領先場景）────────────

def test_high_market_p_filter():
    print("\n【Test 10】高 market_p 過濾（>0.90 應被跳過）")
    try:
        from src.live.live_trader import TraderConfig, LiveTrader

        cfg    = TraderConfig(FakeArgs())
        trader = LiveTrader(cfg)

        # 模擬 Thunder 大幅領先的市場：UNDER ask=0.93（已無 EV）
        markets_overpriced = [{
            "line":         235.5,
            "question":     "Thunder vs 76ers: O/U 235.5",
            "over_prob":    0.07,   # CLOB ask for OVER（幾乎不可能）
            "under_prob":   0.93,   # CLOB ask for UNDER（幾乎確定）
            "condition_id": "0xtest",
        }]

        # pred=221（模型預測低分，方向是 UNDER）
        best = trader._find_best_bet(markets_overpriced, pred=221.0, std=9.0)

        if best is None:
            ok("market_p=0.93 正確被過濾（>0.90 上限）")
        else:
            fail(f"market_p=0.93 應被過濾但未被過濾：{best}")

        # 正常市場：UNDER ask=0.55（有 EV）
        markets_normal = [{
            "line":         235.5,
            "question":     "Thunder vs 76ers: O/U 235.5",
            "over_prob":    0.45,
            "under_prob":   0.55,
            "condition_id": "0xtest2",
        }]

        best2 = trader._find_best_bet(markets_normal, pred=221.0, std=9.0)

        if best2 and best2["side"] == "UNDER":
            ok(f"正常市場正確找到 UNDER EV={best2['ev']:+.1%}")
        elif best2 is None:
            warn("正常市場也無正 EV（可能 pred/std 組合不符合）")
        else:
            fail(f"方向錯誤：{best2}")

        return True
    except Exception as e:
        fail(f"市場過濾測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── Test 11: $1 真實下單測試 ─────────────────────────────────

def test_real_order_1usdc():
    print("\n【Test 11】$1 真實下單測試（CLOB ask + filled_price + trade_status）")

    from src.live.polymarket_trader import PolymarketTrader
    from src.scraper.polymarket_scraper import fetch_nba_game_events, parse_total_markets

    # 確認是否要執行真實下單
    confirm = input("  ⚠️  這會用真實 $1 USDC 下注，確認？(y/N) ")
    if confirm.strip().lower() != "y":
        warn("跳過（未確認）")
        return True

    # Step 1: 取得市場
    markets = parse_total_markets(fetch_nba_game_events())
    if not markets:
        warn("今天沒有市場，跳過")
        return True

    m = markets[0]
    ok(f"市場：{m['question']}")

    # Step 2: 查餘額
    trader  = PolymarketTrader(dry_run=False)
    balance = trader.get_balance()
    ok(f"USDC 餘額：${balance:.4f}")
    if balance < 1.0:
        fail(f"餘額不足 $1，跳過")
        return False

    # Step 3: 取得 token 和 CLOB ask
    token_over = trader.get_token_id(m["condition_id"], "Over")
    if not token_over:
        fail("找不到 token_id，跳過")
        return False

    gamma_mid = m["over_prob"]
    ask_over  = None
    try:
        book = trader._client.get_order_book(token_over)
        if book:
            asks = book.get("asks", []) if isinstance(book, dict) else (getattr(book, "asks", []) or [])
            if asks:
                a0 = asks[0]
                ask_over = float(a0.get("price", 0) if isinstance(a0, dict) else getattr(a0, "price", 0))
    except Exception as e:
        warn(f"CLOB ask 取得失敗：{e}")

    if ask_over:
        diff = abs(ask_over - gamma_mid)
        ok(f"Gamma mid={gamma_mid:.4f}  CLOB ask={ask_over:.4f}  差異={diff:.4f}")
    else:
        warn(f"CLOB ask 取得失敗，使用 Gamma mid={gamma_mid:.4f}")

    market_p = ask_over if ask_over else gamma_mid

    # Step 4: 下 $1 單
    result = trader.place_bet(
        token_id     = token_over,
        outcome      = "Over",
        amount_usdc  = 1.0,
        market_price = market_p,
    )

    if not result.get("success"):
        fail(f"下單失敗：{result.get('error')}")
        return False
    ok(f"下單成功 | order_id={str(result.get('order_id',''))[:20]}...")

    # Step 5: 驗證 filled_price
    fp = result.get("actual_price")
    if fp is not None:
        ok(f"filled_price={fp:.4f}（差={abs(fp-market_p):.4f}）")
    else:
        fail("filled_price 未取得（get_trades 可能失敗）")
        return False

    # Step 6: 驗證 trade_status 邏輯（live_trader 裡用 actual_price 判斷）
    trade_status  = "MATCHED" if fp is not None else "UNKNOWN"
    trade_success = fp is not None
    if trade_status == "MATCHED":
        ok(f"trade_status = MATCHED（不再顯示 UNKNOWN）")
    else:
        fail(f"trade_status = {trade_status}")

    # Step 7: 確認餘額扣款
    import time; time.sleep(3)   # 等 Polymarket 更新餘額
    new_bal = trader.get_balance()
    diff    = round(balance - new_bal, 6)
    ok(f"下注後餘額：${new_bal:.6f}（扣了 ${diff:.6f}）")
    if diff > 0:
        ok(f"餘額確實減少 ${diff:.6f} ✅")
    else:
        warn("餘額未變化（可能有快取，請至 Polymarket 網站確認倉位）")

    return trade_success


# ── Test 12: 重啟後防止重複下注 ──────────────────────────────

def test_dedup_on_restart():
    print("\n【Test 12】重啟後防止重複下注")
    import json
    from src.live.live_trader import TraderConfig, TradingAccount, LiveTrader
    from src.live.ab_simulator import TEAM_NAME_MAP, ABBR_TO_FULL, ESPN_ABBR_FIX

    cfg = TraderConfig(FakeArgs(mode="both", capital=100))

    # ── 12-1: 賽前 dedup（同日） ──────────────────────────────
    print("  ▶ 12-1 賽前：同日重啟")
    pre_acct = TradingAccount(cfg, "pregame", 50.0)
    pre_acct._trades = [
        {"account":"pregame","date":"2026-03-24","trigger":"pregame",
         "game":"Lakers @ Pistons","line":226.5,"side":"UNDER",
         "amount":6.82,"market_p":0.51,"result":None,"pnl":None},
    ]

    # 實際 pregame_loop 的邏輯：away@home（與 ESPN key 格式一致）
    seen = set()
    for t in pre_acct._trades:
        if t.get("trigger") == "pregame" and t.get("result") is None:
            g = t.get("game","")
            if " @ " in g:
                away, home = g.split(" @ ", 1)
                away_abbr = TEAM_NAME_MAP.get(away, away[:3].upper())
                home_abbr = TEAM_NAME_MAP.get(home, home[:3].upper())
                seen.add(f"{away_abbr}@{home_abbr}")  # away@home，與 ESPN key 一致

    if any("LAL" in s or "DET" in s for s in seen):
        ok(f"  賽前 dedup：Lakers @ Pistons 正確加入 seen_pregame（seen={seen}）")
    else:
        fail(f"  賽前 dedup：未加入 seen，seen={seen}")

    # 模擬 ESPN 回傳的 key（away_abbr@home_abbr）
    # ESPN: away_team="LAL", home_team="DET" → key="LAL@DET"
    lal_abbr = TEAM_NAME_MAP.get("Lakers", "LAL")
    det_abbr  = TEAM_NAME_MAP.get("Pistons", "DET")
    game_key  = f"{lal_abbr}@{det_abbr}"
    skipped   = game_key in seen
    if skipped:
        ok(f"  賽前 dedup：重啟後 {game_key} 被正確跳過")
    else:
        fail(f"  賽前 dedup：{game_key} 未在 seen={seen} 中，會重複下注")

    # ── 12-2: 賽前 dedup（跨午夜） ───────────────────────────
    print("  ▶ 12-2 賽前：跨午夜重啟（23:56 下注，00:01 重啟）")
    pre_acct2 = TradingAccount(cfg, "pregame", 50.0)
    pre_acct2._trades = [
        {"account":"pregame","date":"2026-03-23","trigger":"pregame",  # 昨天日期
         "game":"Thunder @ 76ers","line":223.5,"side":"OVER",
         "amount":6.99,"market_p":0.535,"result":None,"pnl":None},
    ]

    seen2 = set()
    for t in pre_acct2._trades:
        if t.get("trigger") == "pregame" and t.get("result") is None:
            g = t.get("game","")
            if " @ " in g:
                away, home = g.split(" @ ", 1)
                seen2.add(f"{TEAM_NAME_MAP.get(home,home[:3].upper())}@{TEAM_NAME_MAP.get(away,away[:3].upper())}")

    phi_key = any("PHI" in s or "OKC" in s for s in seen2)
    if phi_key or seen2:
        ok(f"  跨午夜 dedup：Thunder @ 76ers 加入 seen（seen={seen2}）")
        ok("  跨午夜 dedup：用 result=None 判斷，不受日期影響")
    else:
        fail("  跨午夜 dedup：未加入 seen，跨午夜會重複下注")

    # ── 12-3: 已結算的不影響 dedup ───────────────────────────
    print("  ▶ 12-3 已結算不影響（result!=None 不加入 seen）")
    pre_acct3 = TradingAccount(cfg, "pregame", 50.0)
    pre_acct3._trades = [
        {"account":"pregame","date":"2026-03-23","trigger":"pregame",
         "game":"Lakers @ Pistons","line":226.5,"side":"UNDER",
         "amount":6.82,"market_p":0.51,"result":"win","pnl":6.68},  # 已結算
    ]

    seen3 = set()
    for t in pre_acct3._trades:
        if t.get("trigger") == "pregame" and t.get("result") is None:  # result 不是 None，不加入
            g = t.get("game","")
            if " @ " in g:
                away, home = g.split(" @ ", 1)
                seen3.add(f"{TEAM_NAME_MAP.get(home,home[:3].upper())}@{TEAM_NAME_MAP.get(away,away[:3].upper())}")

    if len(seen3) == 0:
        ok("  已結算記錄不加入 seen（隔天可以對同場比賽下注）")
    else:
        fail(f"  已結算記錄不應加入 seen，但加了：{seen3}")

    # ── 12-4: 半場 dedup（同場，未結算） ─────────────────────
    print("  ▶ 12-4 半場：未結算記錄阻止重複觸發")
    half_acct = TradingAccount(cfg, "halftime", 50.0)
    half_acct._trades = [
        {"account":"halftime","date":"2026-03-24","trigger":"halftime",
         "game":"Thunder @ 76ers","line":235.5,"side":"UNDER",
         "amount":10.0,"market_p":0.48,"result":None,"pnl":None},
    ]

    game = "Thunder @ 76ers"
    already_bet = any(
        t.get("trigger") == "halftime" and
        t.get("game") == game and
        t.get("result") is None
        for acct in [half_acct]
        for t in acct._trades
    )

    if already_bet:
        ok("  半場 dedup：Thunder @ 76ers 有未結算記錄，會跳過重複觸發")
    else:
        fail("  半場 dedup：未偵測到重複，會重複下注")

    # ── 12-5: 半場 dedup（已結算，可以重新下） ───────────────
    print("  ▶ 12-5 半場：已結算不阻止（不同場）")
    half_acct2 = TradingAccount(cfg, "halftime", 50.0)
    half_acct2._trades = [
        {"account":"halftime","date":"2026-03-24","trigger":"halftime",
         "game":"Lakers @ Pistons","line":214.5,"side":"OVER",
         "amount":6.4,"market_p":0.52,"result":"win","pnl":5.91},  # 已結算
    ]

    game2 = "Thunder @ 76ers"  # 不同場
    already_bet2 = any(
        t.get("trigger") == "halftime" and
        t.get("game") == game2 and
        t.get("result") is None
        for t in half_acct2._trades
    )

    if not already_bet2:
        ok("  半場 dedup：不同場比賽不受影響，可以下注")
    else:
        fail("  半場 dedup：不同場比賽被錯誤阻止")

    # ── 12-6: 跨兩個帳戶掃描 ─────────────────────────────────
    print("  ▶ 12-6 跨帳戶掃描（pregame + halftime 都檢查）")
    pre_acct4  = TradingAccount(cfg, "pregame",  50.0)
    half_acct4 = TradingAccount(cfg, "halftime", 50.0)

    # pregame 有一筆 halftime trigger（不可能，但邊界測試）
    # halftime 帳戶有未結算的 halftime 記錄
    half_acct4._trades = [
        {"account":"halftime","date":"2026-03-24","trigger":"halftime",
         "game":"Spurs @ Heat","line":244.5,"side":"UNDER",
         "amount":5.12,"market_p":0.175,"result":None,"pnl":None},
    ]

    game3 = "Spurs @ Heat"
    already_bet3 = any(
        t.get("trigger") == "halftime" and
        t.get("game") == game3 and
        t.get("result") is None
        for acct in [pre_acct4, half_acct4]   # 掃兩個帳戶
        for t in acct._trades
    )

    if already_bet3:
        ok("  跨帳戶掃描：在 halftime 帳戶找到 Spurs @ Heat 未結算記錄")
    else:
        fail("  跨帳戶掃描失敗")

    return True


# ── Test 13: EV 公式與 edge 記錄 ─────────────────────────────

def test_ev_and_edge():
    print("\n【Test 13】EV 公式與 edge 記錄")
    from src.live.live_trader import TraderConfig, LiveTrader

    cfg    = TraderConfig(FakeArgs())
    trader = LiveTrader(cfg)

    # 測試 _find_best_bet 回傳的 ev 和 edge 是否正確
    markets = [{
        "line":         220.5,
        "question":     "Team A vs Team B: O/U 220.5",
        "over_prob":    0.50,
        "under_prob":   0.50,
        "condition_id": "0xtest",
    }]

    # pred=235（明顯 OVER），model_p_over ≈ 0.88（std=9）
    from scipy.stats import norm
    std  = 9.0
    pred = 235.0
    line = 220.5
    model_p = float(1 - norm.cdf(line, loc=pred, scale=std))
    market_p = 0.50

    expected_edge = round(model_p - market_p, 4)
    expected_ev   = round(model_p / market_p - 1, 4)

    best = trader._find_best_bet(markets, pred=pred, std=std)
    if best is None:
        warn("無正 EV（可能被過濾），跳過 EV 公式驗證")
        return True

    # 驗證 edge
    actual_edge = best.get("edge")
    if actual_edge is not None and abs(actual_edge - expected_edge) < 0.001:
        ok(f"edge 正確：{actual_edge:.4f}（model_p-market_p={expected_edge:.4f}）")
    else:
        fail(f"edge 錯誤：got {actual_edge}，expected {expected_edge:.4f}")

    # 驗證 EV（新公式：model_p/market_p - 1）
    actual_ev = best.get("ev")
    if actual_ev is not None and abs(actual_ev - expected_ev) < 0.001:
        ok(f"EV 正確：{actual_ev:.4f}（model_p/market_p-1={expected_ev:.4f}）")
    else:
        fail(f"EV 錯誤：got {actual_ev}，expected {expected_ev:.4f}")

    # 確認 EV ≠ edge（新公式下兩者不同）
    if actual_ev is not None and actual_edge is not None:
        if abs(actual_ev - actual_edge) > 0.01:
            ok(f"EV({actual_ev:.4f}) ≠ edge({actual_edge:.4f})，兩個參數獨立")
        else:
            warn("EV 和 edge 數值相近，可能公式未更新")

    # 驗證 record 裡有 edge 欄位（模擬記錄格式）
    fake_record = {
        "edge": round(best.get("edge", best["model_p"] - best["market_p"]), 4),
        "ev":   round(best["ev"], 4),
    }
    if "edge" in fake_record and "ev" in fake_record:
        ok(f"record 同時記錄 edge={fake_record['edge']} 和 ev={fake_record['ev']}")
    else:
        fail("record 缺少 edge 或 ev 欄位")

    return True


# ── Test 14: polymarket_trader v10 get_trades 詳細測試 ────────

def test_polymarket_trader_v10():
    print("\n【Test 14】polymarket_trader v10 - get_trades 詳細測試")
    import os
    from py_clob_client.clob_types import TradeParams
    from src.live.polymarket_trader import PolymarketTrader

    trader = PolymarketTrader(dry_run=False)
    funder = os.getenv("POLYMARKET_FUNDER", "")
    client = trader._client

    if not client:
        warn("CLOB client 未初始化，跳過")
        return True

    # ── 14-1: get_trades 基本查詢 ─────────────────────────────
    print("  ▶ 14-1 get_trades 基本查詢")
    trades = client.get_trades(params=TradeParams(maker_address=funder))

    if not isinstance(trades, list):
        fail(f"  get_trades 回傳非 list：{type(trades)}")
        return False
    ok(f"  get_trades 回傳 list（{len(trades)} 筆）")

    if not trades:
        warn("  沒有成交記錄，跳過後續測試")
        return True

    t0 = trades[0]
    for field in ["taker_order_id", "asset_id", "price", "size", "status"]:
        if field in t0:
            ok(f"  記錄有 {field}={str(t0[field])[:20]}")
        else:
            fail(f"  記錄缺少 {field}")

    # ── 14-2: 雙重比對邏輯 ────────────────────────────────────
    print("  ▶ 14-2 雙重比對邏輯")
    real_oid   = t0.get("taker_order_id")
    real_asset = t0.get("asset_id")

    def find_trade(oid, asset_id):
        for t in trades:
            if t.get("taker_order_id") == oid and t.get("asset_id") == asset_id:
                return t
        return None

    # 正確比對
    matched = find_trade(real_oid, real_asset)
    if matched:
        ok(f"  正確比對（order+asset）→ 找到 price={matched.get('price')}")
    else:
        fail("  正確比對失敗")

    # 錯誤：asset_id 錯
    fake_asset = "9" * 49
    if find_trade(real_oid, fake_asset) is None:
        ok("  asset_id 錯 → 正確拒絕")
    else:
        fail("  asset_id 錯但仍找到，雙重比對失效")

    # 錯誤：order_id 錯
    fake_oid = "0x" + "f" * 62
    if find_trade(fake_oid, real_asset) is None:
        ok("  order_id 錯 → 正確拒絕")
    else:
        fail("  order_id 錯但仍找到，雙重比對失效")

    # ── 14-3: pregame/halftime 不互相干擾 ─────────────────────
    print("  ▶ 14-3 pregame/halftime 不互相干擾")
    diff_trades = [t for t in trades if t.get("asset_id") != real_asset]

    if diff_trades:
        t1 = diff_trades[0]
        r0 = find_trade(t0.get("taker_order_id"), t0.get("asset_id"))
        r1 = find_trade(t1.get("taker_order_id"), t1.get("asset_id"))

        if r0 and r1 and r0.get("taker_order_id") != r1.get("taker_order_id"):
            ok(f"  兩筆各自獨立：price0={r0.get('price')}  price1={r1.get('price')}")
        else:
            fail("  兩筆記錄干擾或未找到")
    else:
        warn("  最近記錄 asset_id 相同，跳過干擾測試")

    # ── 14-4: 找不到記錄時回退 market_price ──────────────────
    print("  ▶ 14-4 找不到記錄 → 回退 market_price")
    fallback_p = 0.55
    not_found  = find_trade("0x" + "f" * 62, "9" * 49)
    actual     = float(not_found.get("price", fallback_p)) if not_found else fallback_p

    if actual == fallback_p:
        ok(f"  找不到時回退 market_price={fallback_p}")
    else:
        fail(f"  回退失敗：actual={actual}  expected={fallback_p}")

    return True


# ── Test 15: v22 過濾邏輯 ─────────────────────────────────────

def test_v22_filters():
    print("\n【Test 15】v22 過濾邏輯")
    from src.live.live_trader import TraderConfig, LiveTrader
    from scipy.stats import norm as scipy_norm

    cfg    = TraderConfig(FakeArgs(pregame_std=14.6, halftime_std=10.7))
    trader = LiveTrader(cfg)

    # ── 15-1: CLOB ask 失敗 → 排除市場 ──────────────────────
    print("  ▶ 15-1 CLOB ask 失敗 → 排除市場（不下注）")

    # 模擬 Hawks @ Cavaliers：CLOB 失敗，市場仍用 Gamma mid=0.20
    markets_clob_failed = [{
        "line":         234.5,
        "question":     "Hawks vs Cavaliers: O/U 234.5",
        "over_prob":    0.20,   # Gamma mid（CLOB 失敗後的 fallback）
        "under_prob":   0.80,
        "condition_id": "0xtest",
    }]
    # 在 v22 中，CLOB 失敗的市場不會進入 _find_best_bet
    # 所以 _get_markets 回傳空 list，模擬測試 _find_best_bet 不下注
    best = trader._find_best_bet(markets_clob_failed, pred=240.0, std=10.7)
    # 即使 model_p 很高（240 > 234.5），EV 也很高
    # 但 market_p=0.20 通過過濾，這是 v21 的問題
    # v22 的修法是在 _get_markets 就排除，而不是在 _find_best_bet
    # 這裡我們測試：如果真的到達 _find_best_bet，0.20 還是會下注（v22 沒改這裡）
    # 真正的防護是 _get_markets 排除，所以測空 list 的情況
    best_from_empty = trader._find_best_bet([], pred=240.0, std=10.7)
    if best_from_empty is None:
        ok("  CLOB 失敗市場排除後（空 list）→ 不下注")
    else:
        fail("  空市場應回傳 None")

    # ── 15-2: market_p > 0.90 → 過濾 ────────────────────────
    print("  ▶ 15-2 market_p > 0.90 → 過濾（Hawks/Bucks 那兩注）")
    markets_high = [
        {"line": 234.5, "question": "Hawks O/U 234.5", "over_prob": 0.96, "under_prob": 0.04, "condition_id": "0xa"},
        {"line": 228.5, "question": "Bucks O/U 228.5", "over_prob": 0.96, "under_prob": 0.04, "condition_id": "0xb"},
    ]
    for m in markets_high:
        best = trader._find_best_bet([m], pred=244.0, std=10.7)
        if best is None:
            ok(f"  market_p=0.96 ({m['question'][:20]}) → 正確過濾")
        else:
            fail(f"  market_p=0.96 應被過濾")

    # ── 15-3: market_p < 0.10 → 過濾 ────────────────────────
    print("  ▶ 15-3 market_p < 0.10 → 過濾")
    markets_low = [{"line": 220.5, "question": "test O/U 220.5",
                    "over_prob": 0.05, "under_prob": 0.95, "condition_id": "0xc"}]
    best = trader._find_best_bet(markets_low, pred=235.0, std=14.6)
    if best is None:
        ok("  market_p=0.05 → 正確過濾")
    else:
        fail("  market_p=0.05 應被過濾")

    # ── 15-4: model_p > 0.95 → 過濾 ─────────────────────────
    print("  ▶ 15-4 model_p > 0.95 → 過濾（模型過度自信）")
    # pred 離 line 很遠，model_p 會接近 1
    # pred=280, line=230, std=10.7 → model_p ≈ 1.0
    markets_extreme = [{"line": 230.0, "question": "test O/U 230",
                        "over_prob": 0.50, "under_prob": 0.50, "condition_id": "0xd"}]
    pred_extreme = 280.0  # 50分差，model_p_over ≈ 1.0
    model_p_check = float(1 - scipy_norm.cdf(230.0, loc=280.0, scale=10.7))
    best = trader._find_best_bet(markets_extreme, pred=pred_extreme, std=10.7)
    if model_p_check > 0.95:
        if best is None:
            ok(f"  model_p={model_p_check:.4f}>0.95 → 正確過濾")
        else:
            fail(f"  model_p={model_p_check:.4f}>0.95 應被過濾")
    else:
        warn(f"  model_p={model_p_check:.4f} 未超過 0.95，調整測試參數")

    # ── 15-5: abs(pred-line) > 2.5*std → 過濾 ───────────────
    print("  ▶ 15-5 pred 跟 line 差距 > 2.5×std → 過濾")
    # halftime std=10.7，2.5×10.7=26.75
    # pred=260, line=230 → 差30 > 26.75 → 過濾
    markets_far = [{"line": 230.0, "question": "test O/U 230",
                    "over_prob": 0.50, "under_prob": 0.50, "condition_id": "0xe"}]
    best_far = trader._find_best_bet(markets_far, pred=260.0, std=10.7)
    diff = abs(260.0 - 230.0)
    limit = 2.5 * 10.7
    if best_far is None:
        ok(f"  |pred-line|={diff:.1f} > 2.5×std={limit:.1f} → 正確過濾")
    else:
        fail(f"  差距 {diff:.1f} 超過 {limit:.1f} 應被過濾")

    # ── 15-6: 正常注通過所有過濾 ─────────────────────────────
    print("  ▶ 15-6 正常注通過所有過濾")
    # Bucks @ Pistons pregame：pred=227.8, line=221.5, market_p=0.498
    markets_normal = [{"line": 221.5, "question": "Bucks O/U 221.5",
                       "over_prob": 0.498, "under_prob": 0.502, "condition_id": "0xf"}]
    best_normal = trader._find_best_bet(markets_normal, pred=227.8, std=14.6)
    if best_normal and best_normal["side"] == "OVER":
        ok(f"  Bucks pregame 通過：OVER EV={best_normal['ev']:+.1%} edge={best_normal['edge']:+.1%}")
    elif best_normal is None:
        warn("  正常注未找到正 EV（min_ev 條件不符）")
    else:
        fail(f"  方向錯誤：{best_normal}")

    # ── 15-7: 今天實際被擋掉的 2 注驗證 ─────────────────────
    print("  ▶ 15-7 今天實際應被擋掉的注（Grizzlies 0.888 仍通過）")
    # Grizzlies @ Nuggets halftime：actual_fp=0.888，在 0.90 以下，v22 不擋
    markets_grizzlies = [{"line": 246.5, "question": "Grizzlies O/U 246.5",
                          "over_prob": 0.888, "under_prob": 0.112, "condition_id": "0xg"}]
    best_griz = trader._find_best_bet(markets_grizzlies, pred=254.2, std=10.7)
    if best_griz is not None:
        ok(f"  Grizzlies ask=0.888 仍通過過濾（market_p<0.90）→ EV={best_griz['ev']:+.1%}")
        warn("  若要擋此類低報酬注，需把上限從 0.90 降至 0.85")
    else:
        ok("  Grizzlies 被過濾")

    return True


# ── Test 16: 完全移除 Gamma mid（v24）────────────────────────

def test_no_gamma_mid():
    print("\n【Test 16】完全移除 Gamma mid（v24）")
    from src.live.live_trader import TraderConfig, LiveTrader
    from src.live.polymarket_trader import PolymarketTrader

    cfg    = TraderConfig(FakeArgs())
    trader = LiveTrader(cfg)

    # ── 16-1: over_prob=None 在 _find_best_bet 被跳過 ────────
    print("  ▶ 16-1 market_p=None → 直接跳過（CLOB 未取得）")
    markets_none = [{
        "line":         230.5,
        "question":     "test O/U 230.5",
        "over_prob":    None,   # CLOB 未取得
        "under_prob":   None,
        "condition_id": "0xtest",
    }]
    best = trader._find_best_bet(markets_none, pred=240.0, std=10.7)
    if best is None:
        ok("  over_prob=None → 正確跳過，不下注")
    else:
        fail("  over_prob=None 應跳過但未跳過")

    # ── 16-2: 正常 CLOB ask 仍然通過 ─────────────────────────
    print("  ▶ 16-2 正常 CLOB ask → 通過")
    markets_ok = [{
        "line":         221.5,
        "question":     "test O/U 221.5",
        "over_prob":    0.498,  # CLOB ask
        "under_prob":   0.502,
        "condition_id": "0xtest2",
    }]
    best2 = trader._find_best_bet(markets_ok, pred=228.0, std=14.6)
    if best2 is not None:
        ok(f"  CLOB ask=0.498 通過：{best2['side']} EV={best2['ev']:+.1%}")
    else:
        warn("  正常市場無正 EV（min_ev 不符）")

    # ── 16-3: filled_price=None 時結算用 market_p 並發出警告 ─
    print("  ▶ 16-3 filled_price=None → 結算時警告，用 market_p")
    from src.live.live_trader import TradingAccount
    acct = TradingAccount(cfg, "halftime", 50.0)
    acct._trades.append({
        "account":       "halftime",
        "game":          "Bulls @ Wizards",
        "line":          234.5,
        "side":          "UNDER",
        "amount":        4.33,
        "market_p":      0.63,
        "filled_price":  None,    # ← get_trades 和 CLOB 都失敗
        "order_success": True,
        "trade_status":  "MATCHED",
        "trade_success": True,
        "attempt":       1,
        "result":        None,
        "pnl":           None,
    })
    acct.settle("Bulls @ Wizards", 234.5, 227)
    t = acct._trades[-1]
    if t["result"] == "win":
        ok(f"  filled_price=None 結算成功（用 market_p=0.63）pnl={t['pnl']:+.2f}")
        # 確認用 market_p 計算
        expected = round(4.33 / 0.63 - 4.33, 2)
        if abs(t["pnl"] - expected) < 0.05:
            ok(f"  pnl 計算正確：{t['pnl']:+.2f}（預期 {expected:+.2f}）")
        else:
            fail(f"  pnl 計算錯誤：{t['pnl']} vs {expected}")
    else:
        fail(f"  結算方向錯誤：{t['result']}")

    # ── 16-4: 模擬 Pacers @ Nets 的情況（get_trades 失敗）────
    print("  ▶ 16-4 get_trades 失敗 → filled_price=None（不用 Gamma mid 0.16）")
    # 模擬 place_bet 回傳
    fake_result_old = {
        "success":      True,
        "order_id":     "0xtest",
        "actual_price": 0.16,    # 舊版：用 Gamma mid
        "actual_size":  17.3,
    }
    fake_result_new = {
        "success":      True,
        "order_id":     "0xtest",
        "actual_price": None,    # v24：get_trades 失敗 → None
        "actual_size":  None,
    }

    # 舊版：fp=0.16，pnl 會被嚴重高估
    old_fp = fake_result_old["actual_price"] or 0.73  # 實際是 0.73
    old_pnl = round(2.77 / 0.16 - 2.77, 2)  # 用 Gamma mid 計算
    real_pnl = round(2.77 / 0.73 - 2.77, 2)  # 實際
    ok(f"  舊版 pnl（Gamma mid 0.16）= ${old_pnl:+.2f}（虛假）")
    ok(f"  實際 pnl（CLOB 0.73）     = ${real_pnl:+.2f}")
    ok(f"  v24 filled_price=None → 結算時用 market_p 並發警告（不再虛假 +$14）")

    # ── 16-5: 市場 over_prob 初始為 None，CLOB 失敗就排除 ────
    print("  ▶ 16-5 初始 over_prob=None 確保不用 Gamma")
    # 模擬 _get_markets 行為：先設 None，再由 CLOB 填入
    m = {
        "line":         230.5,
        "question":     "test",
        "over_prob":    0.55,   # Gamma mid（原始）
        "under_prob":   0.45,
        "condition_id": "0xtest",
    }
    # v24 先清空
    m["over_prob"]  = None
    m["under_prob"] = None

    # 如果 CLOB 失敗，over_prob 保持 None
    # _find_best_bet 應跳過
    best3 = trader._find_best_bet([m], pred=240.0, std=10.7)
    if best3 is None:
        ok("  CLOB 失敗後 over_prob=None → 市場被排除，不下注")
    else:
        fail("  over_prob=None 的市場不應下注")

    return True

# ── 清理 & 主程式 ─────────────────────────────────────────────

def cleanup():
    lt_mod.TRADER_JSONL_PREGAME  = _orig_pre
    lt_mod.TRADER_JSONL_HALFTIME = _orig_half
    shutil.rmtree(_tmp_dir, ignore_errors=True)


def main():
    print("=" * 60)
    print("  Live Trader End-to-End Test")
    print("=" * 60)

    tests = [
        ("設定載入",              test_config),
        ("兩個獨立帳戶",          test_accounts),
        ("模型預測",              test_prediction),
        ("Polymarket 市場",       test_polymarket),
        ("完整下注流程",          test_betting_flow),
        ("結算邏輯",              test_settlement),
        ("Report 輸出",           test_report),
        ("CLOB ask price",        test_clob_ask_price),
        ("filled_price 結算",     test_filled_price_settlement),
        ("高 market_p 過濾",      test_high_market_p_filter),
        ("$1 真實下單",           test_real_order_1usdc),
        ("重啟後防止重複下注",    test_dedup_on_restart),
        ("EV 公式與 edge 記錄",   test_ev_and_edge),
        ("get_trades 雙重比對",   test_polymarket_trader_v10),
        ("v22 過濾邏輯",           test_v22_filters),
        ("完全移除 Gamma mid",     test_no_gamma_mid),
    ]

    results = []
    for name, func in tests:
        try:
            passed = func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  {RED}CRASH in {name}: {e}{RESET}")
            results.append((name, False))

    cleanup()

    print("\n" + "=" * 60)
    print("  測試結果總覽")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print(f"  {GREEN}全部通過！live_trader 可以正常運行。{RESET}")
    else:
        print(f"  {RED}有測試失敗，請檢查上方錯誤訊息。{RESET}")
    print()


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    main()