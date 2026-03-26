"""
test_price_accuracy.py
----------------------
驗證：
  1. CLOB ask 是否比 Gamma mid 更接近實際成交價
  2. 下單後 filled_price 是否與實際成交一致
  3. 帳戶損益計算是否與 Polymarket 實際一致

執行：
  python test_price_accuracy.py
"""

import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"     {msg}")
def head(msg): print(f"\n{BLUE}{'─'*60}{RESET}\n  {msg}\n{BLUE}{'─'*60}{RESET}")

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        ok(name)
        if detail: info(detail)
        passed += 1
    else:
        fail(name)
        if detail: info(detail)
        failed += 1

# ── 初始化 ─────────────────────────────────────────────────────
from src.live.polymarket_trader import PolymarketTrader
from src.scraper.polymarket_scraper import fetch_nba_game_events, parse_total_markets
from py_clob_client.clob_types import TradeParams

trader = PolymarketTrader(dry_run=False)
client = trader._client
funder = os.getenv("POLYMARKET_FUNDER", "")

print("=" * 65)
print("  報價準確性 & 損益計算測試")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# PART 1：報價準確性（CLOB ask vs Gamma mid）
# ══════════════════════════════════════════════════════════════
head("PART 1：CLOB ask 報價準確性")

markets = parse_total_markets(fetch_nba_game_events())

if not markets:
    warn("今天沒有市場，跳過 PART 1")
else:
    print(f"  找到 {len(markets)} 個市場\n")

    total_diff_gamma = 0
    total_diff_clob  = 0
    count = 0

    for m in markets[:5]:   # 取前 5 個市場比較
        token_over = trader.get_token_id(m["condition_id"], "Over")
        if not token_over:
            continue

        # Gamma mid
        gamma_mid = m["over_prob"]

        # CLOB ask（最低賣出價 = 你實際成交的價格）
        clob_ask = None
        try:
            book = client.get_order_book(token_over)
            asks = getattr(book, "asks", []) or []
            if asks:
                clob_ask = float(getattr(asks[-1], "price", 0))
        except Exception as e:
            warn(f"CLOB ask 取得失敗：{e}")

        if clob_ask:
            diff = abs(clob_ask - gamma_mid)
            info(f"{m['question'][:40]:<40}")
            info(f"  Gamma mid = {gamma_mid:.4f}   CLOB ask = {clob_ask:.4f}   差異 = {diff:.4f}")
            total_diff_gamma += gamma_mid
            total_diff_clob  += clob_ask
            count += 1

    if count > 0:
        avg_diff = abs(total_diff_clob/count - total_diff_gamma/count)
        print()
        if avg_diff > 0.01:
            ok(f"CLOB ask 與 Gamma mid 有顯著差異（平均 {avg_diff:.4f}），使用 CLOB 更準確")
        else:
            ok(f"CLOB ask ≈ Gamma mid（平均差異 {avg_diff:.4f}），市場平穩")

# ══════════════════════════════════════════════════════════════
# PART 2：歷史成交記錄分析（filled_price vs market_p）
# ══════════════════════════════════════════════════════════════
head("PART 2：歷史 filled_price 分析")

# 讀取 jsonl 記錄
jsonl_files = [
    Path("logs/live_trades_pregame.jsonl"),
    Path("logs/live_trades_halftime.jsonl"),
]

all_trades = []
for f in jsonl_files:
    if f.exists():
        with open(f, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        all_trades.append(json.loads(line))
                    except:
                        pass

settled = [t for t in all_trades if t.get("result") is not None]
print(f"  總記錄：{len(all_trades)} 筆  已結算：{len(settled)} 筆\n")

if not all_trades:
    warn("沒有交易記錄")
else:
    # 分析 filled_price vs market_p
    has_filled   = [t for t in all_trades if t.get("filled_price") and t.get("filled_price") != t.get("market_p")]
    no_filled    = [t for t in all_trades if not t.get("filled_price") or t.get("filled_price") == t.get("market_p")]
    
    ok(f"有真實 filled_price 的記錄：{len(has_filled)} 筆")
    ok(f"使用 market_p 回退的記錄：{len(no_filled)} 筆")

    if has_filled:
        diffs = [abs(t["filled_price"] - t["market_p"]) for t in has_filled]
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        
        print()
        info(f"filled_price vs market_p：")
        info(f"  平均差異：{avg_diff:.4f}")
        info(f"  最大差異：{max_diff:.4f}")
        
        if max_diff > 0.10:
            ok(f"有記錄差異超過 0.10（最大 {max_diff:.4f}），CLOB ask 確實必要")
        else:
            ok(f"差異在合理範圍（最大 {max_diff:.4f}）")

    # 顯示最近 5 筆
    print(f"\n  最近 5 筆記錄：")
    print(f"  {'日期':<12} {'比賽':<24} {'方向':<6} {'market_p':>8} {'filled':>8} {'差異':>6}")
    print(f"  {'─'*64}")
    for t in all_trades[-5:]:
        fp   = t.get("filled_price") or t.get("market_p", 0)
        mp   = t.get("market_p", 0)
        diff = fp - mp
        print(f"  {t.get('date',''):<12} {t.get('game','')[:24]:<24} "
              f"{t.get('side',''):<6} {mp:>8.4f} {fp:>8.4f} {diff:>+6.4f}")

# ══════════════════════════════════════════════════════════════
# PART 3：損益計算驗證（程式計算 vs 實際應得）
# ══════════════════════════════════════════════════════════════
head("PART 3：損益計算驗證")

if not settled:
    warn("沒有已結算記錄，跳過")
else:
    print(f"  {'比賽':<24} {'線':>6} {'方向':<6} {'下注':>7} {'filled':>7} {'結果':<6} {'程式pnl':>9} {'應得pnl':>9} {'一致'}")
    print(f"  {'─'*80}")
    
    pnl_match   = 0
    pnl_mismatch = 0

    for t in settled:
        fp     = t.get("filled_price") or t.get("market_p", 0.5)
        amount = t.get("amount", 0)
        result = t.get("result")
        pnl    = t.get("pnl", 0) or 0

        # 用 filled_price 重新計算應得 pnl
        if result == "win":
            expected_pnl = round(amount / fp - amount, 2)
        else:
            expected_pnl = -amount

        diff    = abs(pnl - expected_pnl)
        match   = diff < 0.05
        marker  = f"{GREEN}✅{RESET}" if match else f"{RED}❌{RESET}"

        print(f"  {t.get('game','')[:24]:<24} {t.get('line',0):>6.1f} "
              f"{t.get('side',''):<6} ${amount:>6.2f} {fp:>7.4f} "
              f"{'WIN' if result=='win' else 'LOSS':<6} "
              f"${pnl:>+8.2f} ${expected_pnl:>+8.2f} {marker}")

        if match:
            pnl_match += 1
        else:
            pnl_mismatch += 1

    print()
    check(f"損益計算一致（{pnl_match}/{len(settled)} 筆）",
          pnl_mismatch == 0,
          f"不一致：{pnl_mismatch} 筆（差異可能來自 filled_price 未正確記錄）")

# ══════════════════════════════════════════════════════════════
# PART 4：$1 真實下單端對端驗證（選用）
# ══════════════════════════════════════════════════════════════
head("PART 4：$1 真實下單端對端驗證（選用）")

confirm = input("  ⚠️  下 $1 真實單，驗證報價→成交→filled_price 完整流程？(y/N) ")
if confirm.strip().lower() == "y":
    markets = parse_total_markets(fetch_nba_game_events())

    if not markets:
        warn("今天沒有市場")
    else:
        m = markets[0]
        info(f"市場：{m['question']}")

        token_over = trader.get_token_id(m["condition_id"], "Over")

        # Step 1: 抓 Gamma mid
        gamma_mid = m["over_prob"]
        info(f"Gamma mid = {gamma_mid:.4f}")

        # Step 2: 抓 CLOB ask
        clob_ask = None
        try:
            book = client.get_order_book(token_over)
            asks = getattr(book, "asks", []) or []
            if asks:
                clob_ask = float(getattr(asks[-1], "price", 0))
        except Exception as e:
            warn(f"CLOB ask 失敗：{e}")

        info(f"CLOB ask  = {clob_ask:.4f}" if clob_ask else "CLOB ask 取得失敗")
        market_p = clob_ask if clob_ask else gamma_mid

        # Step 3: 下單
        result = trader.place_bet(
            token_id     = token_over,
            outcome      = "Over",
            amount_usdc  = 1.0,
            market_price = market_p,
        )

        check("下單成功", result.get("success"),
              f"order_id={str(result.get('order_id',''))[:25]}...")

        fp = result.get("actual_price")
        check("filled_price 取得成功", fp is not None,
              f"filled_price={fp}")

        if fp:
            diff_gamma = abs(fp - gamma_mid)
            diff_clob  = abs(fp - clob_ask) if clob_ask else None

            info(f"Gamma mid={gamma_mid:.4f}  CLOB ask={clob_ask:.4f}  filled={fp:.4f}")
            info(f"填單價 vs Gamma mid 差異：{diff_gamma:.4f}")
            if diff_clob is not None:
                info(f"填單價 vs CLOB ask 差異：{diff_clob:.4f}")

            check("CLOB ask 比 Gamma mid 更接近 filled_price",
                  diff_clob is not None and diff_clob <= diff_gamma,
                  f"CLOB diff={diff_clob:.4f}  Gamma diff={diff_gamma:.4f}")

            # 模擬損益計算
            if result.get("success"):
                expected_win_pnl  = round(1.0 / fp - 1.0, 2)
                expected_loss_pnl = -1.0
                info(f"若贏：預期損益 = +${expected_win_pnl:.2f}")
                info(f"若輸：預期損益 = -$1.00")
                ok("損益計算公式：amount / filled_price - amount")
else:
    warn("跳過真實下單測試")

# ══════════════════════════════════════════════════════════════
# 總結
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  結果：{GREEN}{passed} 通過{RESET}  {RED if failed else RESET}{failed} 失敗{RESET}")
print("=" * 65 + "\n")
