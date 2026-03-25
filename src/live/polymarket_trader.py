import os
"""
polymarket_trader.py
--------------------
透過 Polymarket CLOB API 實際下注。

前置需求：
  pip install py-clob-client python-dotenv

.env 設定：
  POLYMARKET_PRIVATE_KEY=0x...        # 從 Polymarket 匯出的私鑰
  POLYMARKET_FUNDER=0x...             # polymarket.com/settings 顯示的錢包地址（Proxy Wallet）
  POLYMARKET_API_KEY=...              # 產生後填入
  POLYMARKET_API_SECRET=...
  POLYMARKET_API_PASSPHRASE=...

使用方式：
  python src/live/polymarket_trader.py --setup-key
  python src/live/polymarket_trader.py --dry-run
  python src/live/polymarket_trader.py --balance
  python src/live/polymarket_trader.py --test-order
"""

import sys
import logging
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

log = logging.getLogger(__name__)

POLYGON_CHAIN_ID = 137
CLOB_HOST        = "https://clob.polymarket.com"
SIGNATURE_TYPE   = 1        # POLY_PROXY (Polymarket 內建錢包匯出的私鑰)
MIN_BET_USDC     = 1.0
MAX_BET_USDC     = 500.0


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent.parent / ".env")
    except ImportError:
        pass
    return {
        "private_key": os.getenv("POLYMARKET_PRIVATE_KEY"),
        "funder":      os.getenv("POLYMARKET_FUNDER"),
        "api_key":     os.getenv("POLYMARKET_API_KEY"),
        "api_secret":  os.getenv("POLYMARKET_API_SECRET"),
        "api_pass":    os.getenv("POLYMARKET_API_PASSPHRASE"),
    }


def setup_api_key():
    """用私鑰產生 / 取得 API 憑證（create_or_derive）"""
    env = _load_env()

    if not env["private_key"]:
        print("❌ 缺少 POLYMARKET_PRIVATE_KEY")
        return
    if not env["funder"]:
        print("❌ 缺少 POLYMARKET_FUNDER")
        print("   請到 polymarket.com/settings 查看錢包地址，填入 .env：")
        print("   POLYMARKET_FUNDER=0x...")
        return

    try:
        from py_clob_client.client import ClobClient
        client = ClobClient(
            host           = CLOB_HOST,
            key            = env["private_key"],
            chain_id       = POLYGON_CHAIN_ID,
            signature_type = SIGNATURE_TYPE,
            funder         = env["funder"],
        )
        print("正在產生 / 取得 API 憑證...")
        creds = client.create_or_derive_api_creds()
        print("\n✅ 成功！請將以下內容加入 .env：\n")
        print(f"POLYMARKET_API_KEY={creds.api_key}")
        print(f"POLYMARKET_API_SECRET={creds.api_secret}")
        print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
        print("\n⚠️  請立刻複製儲存")
    except Exception as e:
        print(f"❌ 失敗：{e}")
        print("\n可能原因：")
        print("  1. POLYMARKET_FUNDER 地址不正確（請確認是 polymarket.com/settings 的地址）")
        print("  2. 私鑰格式錯誤（需要 0x 開頭）")


class PolymarketTrader:

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._client = None
        self._init_client()

    def _init_client(self):
        env = _load_env()
        if not env["private_key"] or not env["funder"]:
            log.error("缺少 POLYMARKET_PRIVATE_KEY 或 POLYMARKET_FUNDER")
            return
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            kwargs = dict(
                host           = CLOB_HOST,
                key            = env["private_key"],
                chain_id       = POLYGON_CHAIN_ID,
                signature_type = SIGNATURE_TYPE,
                funder         = env["funder"],
            )
            if env["api_key"] and env["api_secret"] and env["api_pass"]:
                kwargs["creds"] = ApiCreds(
                    api_key        = env["api_key"],
                    api_secret     = env["api_secret"],
                    api_passphrase = env["api_pass"],
                )
            self._client = ClobClient(**kwargs)
            log.info("Polymarket CLOB client 初始化成功")
        except ImportError:
            log.error("缺少 py-clob-client：pip install py-clob-client")
        except Exception as e:
            log.error(f"初始化失敗：{e}")

    def get_balance(self) -> float:
        if not self._client:
            return 0.0
        try:
            # get_balance_allowance 回傳 {asset: "USDC", balance: "...", allowance: "..."}
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            resp   = self._client.get_balance_allowance(params=params)
            # balance 單位是 USDC 6位小數（例如 9993301 = $9.993301）
            raw  = resp.get("balance", "0") if isinstance(resp, dict) else str(resp)
            usdc = float(raw) / 1e6
            log.info(f"USDC 餘額：${usdc:,.4f}")
            return usdc
        except Exception as e:
            log.error(f"查詢餘額失敗：{e}")
            return 0.0

    def get_token_id(self, condition_id: str, outcome: str) -> str | None:
        """從 condition_id + outcome 取得 token_id"""
        if not self._client:
            return None
        try:
            market = self._client.get_market(condition_id)
            for token in market.get("tokens", []):
                if token.get("outcome", "").lower() == outcome.lower():
                    return token.get("token_id")
            log.error(f"找不到 outcome={outcome!r} 的 token")
            return None
        except Exception as e:
            log.error(f"get_token_id 失敗：{e}")
            return None

    def place_bet(
        self,
        token_id:     str,
        outcome:      str,
        amount_usdc:  float,
        market_price: float,
        tick_size:    str  = "0.01",
        neg_risk:     bool = False,
    ) -> dict:
        result = {
            "success":   False,
            "order_id":  None,
            "shares":    0.0,
            "cost_usdc": amount_usdc,
            "dry_run":   self.dry_run,
            "error":     None,
        }

        if amount_usdc < MIN_BET_USDC:
            result["error"] = f"金額 ${amount_usdc:.2f} < 最小值 ${MIN_BET_USDC}"
            log.warning(result["error"])
            return result

        if amount_usdc > MAX_BET_USDC:
            log.warning(f"截斷：${amount_usdc:.2f} → ${MAX_BET_USDC}")
            amount_usdc = MAX_BET_USDC
            result["cost_usdc"] = MAX_BET_USDC

        shares = round(amount_usdc / market_price, 2)
        result["shares"] = shares

        if self.dry_run:
            log.info(
                f"[DRY RUN] {outcome} | token={token_id[:10]}... | "
                f"${amount_usdc:.2f} @ {market_price:.3f} = {shares:.1f} 股"
            )
            result["success"]  = True
            result["order_id"] = "DRY_RUN"
            return result

        if not self._client:
            result["error"] = "CLOB client 未初始化"
            return result

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            # 使用 FOK 市價單：amount 直接是 USDC 金額
            mo     = MarketOrderArgs(token_id=token_id, amount=amount_usdc, side=BUY)
            signed = self._client.create_market_order(mo)
            resp   = self._client.post_order(signed, OrderType.FOK)

            if resp.get("success"):
                oid = resp.get("orderID", "")
                log.info(f"下單成功 ✅ {outcome} | ${amount_usdc:.2f} FOK | order={oid}")
                result["success"]  = True
                result["order_id"] = oid

                # 查詢實際成交價格
                try:
                    from py_clob_client.clob_types import TradeParams
                    # 用 maker_address 查，然後比對 taker_order_id + asset_id
                    funder = os.getenv("POLYMARKET_FUNDER", "")
                    trades = self._client.get_trades(
                        params=TradeParams(maker_address=funder)
                    )
                    # 雙重比對：taker_order_id（order id）+ asset_id（token id）
                    matched_trade = None
                    if trades and isinstance(trades, list):
                        for t in trades:
                            if (t.get("taker_order_id") == oid and
                                t.get("asset_id") == token_id):
                                matched_trade = t
                                break
                    if matched_trade:
                        actual_price = float(matched_trade.get("price", market_price))
                        actual_size  = float(matched_trade.get("size",  shares))
                        result["actual_price"] = actual_price
                        result["actual_size"]  = actual_size
                        log.info(f"  實際成交：price={actual_price:.4f}  size={actual_size:.4f}")
                    else:
                        result["actual_price"] = market_price
                        result["actual_size"]  = shares
                        log.warning(f"  找不到成交記錄，使用 market_price={market_price:.4f}")
                except Exception as e:
                    log.warning(f"  查詢成交價失敗：{e}")
                    result["actual_price"] = market_price
                    result["actual_size"]  = shares
            else:
                result["error"] = resp.get("errorMsg", str(resp))
                log.error(f"下單失敗：{result['error']}")
        except Exception as e:
            result["error"] = str(e)
            log.error(f"下單例外：{e}")

        return result

    def cancel_order(self, order_id: str) -> bool:
        if self.dry_run or not self._client:
            return True
        try:
            return self._client.cancel(order_id).get("canceled", False)
        except Exception as e:
            log.error(f"取消失敗：{e}")
            return False


def diagnose():
    print("\n" + "="*55)
    print("  Polymarket Trader 診斷")
    print("="*55)

    try:
        import py_clob_client
        print("✅ py-clob-client 已安裝")
    except ImportError:
        print("❌ py-clob-client 未安裝 → pip install py-clob-client")
        return

    env = _load_env()
    for name, val, req in [
        ("POLYMARKET_PRIVATE_KEY",    env["private_key"], True),
        ("POLYMARKET_FUNDER",         env["funder"],      True),
        ("POLYMARKET_API_KEY",        env["api_key"],     False),
        ("POLYMARKET_API_SECRET",     env["api_secret"],  False),
        ("POLYMARKET_API_PASSPHRASE", env["api_pass"],    False),
    ]:
        if val:
            masked = val[:6] + "..." + val[-4:] if len(val) > 10 else "***"
            print(f"✅ {name} = {masked}")
        elif req:
            print(f"❌ {name} 未設定（必填）")
        else:
            print(f"⚠️  {name} 未設定（先執行 --setup-key）")

    print("\n  測試連線...")
    trader = PolymarketTrader(dry_run=True)
    if not trader._client:
        print("❌ CLOB client 初始化失敗")
        return
    print("✅ CLOB client 連線成功")

    if env["api_key"]:
        bal = trader.get_balance()
        if bal >= 0:
            print(f"✅ USDC 餘額：${bal:,.4f}")
    else:
        print("⚠️  跳過餘額查詢（需要 API Key）")

    print("\n  Dry-run 下注測試...")
    r = trader.place_bet("test_token_000", "Over", 5.0, 0.52)
    if r["success"]:
        print(f"✅ Dry-run 成功：{r['shares']:.1f} 股 @ $0.52")
    else:
        print(f"❌ Dry-run 失敗：{r['error']}")

    print("="*55 + "\n")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-key",  action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--balance",    action="store_true")
    parser.add_argument("--test-order", action="store_true")
    args = parser.parse_args()

    if args.setup_key:
        setup_api_key()
    elif args.balance:
        PolymarketTrader(dry_run=False).get_balance()
    elif args.test_order:
        print("⚠️  這會用真實 USDC 下注 $1，確認？(y/N)")
        if input().strip().lower() == "y":
            from src.scraper.polymarket_scraper import fetch_nba_game_events, parse_total_markets
            markets = parse_total_markets(fetch_nba_game_events())
            if not markets:
                print("找不到市場")
            else:
                m = markets[0]
                print(f"市場：{m['question']}  Over={m['over_prob']:.2f}")
                trader   = PolymarketTrader(dry_run=False)
                token_id = trader.get_token_id(m["condition_id"], "Over")
                if token_id:
                    print(trader.place_bet(token_id, "Over", 1.0, m["over_prob"]))
                else:
                    print("找不到 token_id")
    else:
        diagnose()