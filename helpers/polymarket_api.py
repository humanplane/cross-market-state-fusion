"""
Polymarket API helpers for 15-min up/down markets.
Finds BTC, ETH, SOL, XRP markets using slug pattern.

Uses aiohttp for async requests to avoid blocking the event loop.
"""
import aiohttp
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional, Dict

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# 15-min assets (slug pattern: {asset}-updown-15m-{timestamp})
ASSETS_15M = ["btc", "eth", "sol", "xrp"]

# Module-level session for reuse
_session: Optional[aiohttp.ClientSession] = None


@dataclass
class Market:
    """Active 15-min prediction market."""
    condition_id: str
    question: str
    asset: str
    end_time: datetime
    token_up: str
    token_down: str
    price_up: float = 0.5
    price_down: float = 0.5
    slug: str = ""


def _get_session() -> aiohttp.ClientSession:
    """Get or create aiohttp session."""
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=10)
        _session = aiohttp.ClientSession(timeout=timeout)
    return _session


async def get_market_from_clob_async(condition_id: str, session: aiohttp.ClientSession = None) -> Optional[Dict]:
    """Get market details from CLOB API including token IDs (async)."""
    url = f"{CLOB_API}/markets/{condition_id}"
    try:
        sess = session or _get_session()
        async with sess.get(url) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except Exception:
        return None


async def _fetch_market_for_asset(session: aiohttp.ClientSession, asset: str, ts: int, now: datetime) -> Optional[Market]:
    """Fetch a single market for an asset/timestamp combination."""
    slug = f"{asset}-updown-15m-{ts}"
    url = f"{GAMMA_API}/events?slug={slug}"

    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None

            events = await resp.json()
            if not events:
                return None

            e = events[0]
            end_str = e.get("endDate", "")

            if not end_str:
                return None

            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

            if end_dt <= now:
                return None

            # Get condition ID
            condition_id = None
            for m in e.get("markets", []):
                condition_id = m.get("conditionId")
                if condition_id:
                    break

            if not condition_id:
                return None

            # Get token IDs from CLOB (pass session to avoid event loop issues)
            clob_data = await get_market_from_clob_async(condition_id, session)
            if not clob_data:
                return None

            if not clob_data.get("active") or clob_data.get("closed"):
                return None

            tokens = clob_data.get("tokens", [])
            token_up = None
            token_down = None
            price_up = 0.5
            price_down = 0.5

            for t in tokens:
                outcome = t.get("outcome", "").lower()
                if outcome == "up":
                    token_up = t.get("token_id")
                    price_up = t.get("price", 0.5)
                elif outcome == "down":
                    token_down = t.get("token_id")
                    price_down = t.get("price", 0.5)

            if not token_up or not token_down:
                return None

            return Market(
                condition_id=condition_id,
                question=clob_data.get("question", ""),
                asset=asset.upper(),
                end_time=end_dt,
                token_up=token_up,
                token_down=token_down,
                price_up=price_up,
                price_down=price_down,
                slug=slug,
            )
    except Exception:
        return None


async def _get_15m_markets_with_session(session: aiohttp.ClientSession, assets: List[str] = None) -> List[Market]:
    """
    Core implementation that fetches markets using provided session.

    Args:
        session: aiohttp session to use for requests
        assets: List of assets (default: btc, eth, sol, xrp)

    Returns:
        List of active Market objects sorted by end time.
    """
    if assets is None:
        assets = ASSETS_15M
    else:
        assets = [a.lower() for a in assets]

    now = datetime.now(timezone.utc)
    current_ts = int(now.timestamp())

    # Round to 15-min boundary (900 seconds)
    window_start = (current_ts // 900) * 900

    # Check current and next 3 windows
    timestamps = [window_start + (i * 900) for i in range(4)]

    # Create all fetch tasks
    tasks = []
    for asset in assets:
        for ts in timestamps:
            tasks.append(_fetch_market_for_asset(session, asset, ts, now))

    # Run all requests concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect valid markets (first valid per asset)
    markets = []
    seen_assets = set()

    for i, result in enumerate(results):
        if isinstance(result, Market) and result is not None:
            if result.asset not in seen_assets:
                markets.append(result)
                seen_assets.add(result.asset)

    # Sort by end time
    markets.sort(key=lambda m: m.end_time)
    return markets


async def get_15m_markets_async(assets: List[str] = None) -> List[Market]:
    """
    Get currently active 15-min up/down markets (async).

    Uses slug pattern: {asset}-updown-15m-{timestamp}

    Args:
        assets: List of assets (default: btc, eth, sol, xrp)

    Returns:
        List of active Market objects sorted by end time.
    """
    session = _get_session()
    return await _get_15m_markets_with_session(session, assets)


def get_15m_markets(assets: List[str] = None) -> List[Market]:
    """
    Synchronous wrapper for get_15m_markets_async.

    Creates a fresh session each time to avoid cross-event-loop issues.
    """
    async def _fetch_with_fresh_session():
        # Create a fresh session for this call to avoid event loop issues
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await _get_15m_markets_with_session(session, assets)

    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - run in executor with fresh session
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _fetch_with_fresh_session())
            return future.result()
    except RuntimeError:
        # No running loop - we can create one
        return asyncio.run(_fetch_with_fresh_session())


def get_market_from_clob(condition_id: str) -> Optional[Dict]:
    """Synchronous wrapper for get_market_from_clob_async."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, get_market_from_clob_async(condition_id))
            return future.result()
    except RuntimeError:
        return asyncio.run(get_market_from_clob_async(condition_id))


async def get_next_market_async(asset: str) -> Optional[Market]:
    """Get the next closing 15-min market for a specific asset (async)."""
    markets = await get_15m_markets_async(assets=[asset])
    return markets[0] if markets else None


def get_next_market(asset: str) -> Optional[Market]:
    """Get the next closing 15-min market for a specific asset."""
    markets = get_15m_markets(assets=[asset])
    return markets[0] if markets else None


# Backwards compat
get_active_markets = get_15m_markets


async def close_session():
    """Close the aiohttp session."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


if __name__ == "__main__":
    print("=" * 60)
    print("15-MIN UP/DOWN MARKETS")
    print("=" * 60)

    markets = get_15m_markets()
    now = datetime.now(timezone.utc)

    if not markets:
        print("\nNo active 15-min markets found!")
    else:
        for m in markets:
            mins_left = (m.end_time - now).total_seconds() / 60
            print(f"\n{m.asset} 15m")
            print(f"  {m.question}")
            print(f"  Closes in: {mins_left:.1f} min")
            print(f"  UP: {m.price_up:.3f} | DOWN: {m.price_down:.3f}")
            print(f"  Condition: {m.condition_id}")
            print(f"  Token UP: {m.token_up}")
            print(f"  Token DOWN: {m.token_down}")
