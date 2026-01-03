"""
Live execution adapter connecting RL agent to Polymarket.

This module bridges the paper trading RL agent with real Polymarket execution
using py-clob-client directly.
"""
import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

# Add parent Parallax project to path for imports
PARALLAX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PARALLAX_PATH not in sys.path:
    sys.path.insert(0, PARALLAX_PATH)

# Load environment from Parallax .env manually (dotenv has issues in some contexts)
def load_env_file(path: str):
    """Load .env file manually."""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"').strip("'")
                    if key not in os.environ or not os.environ[key]:
                        os.environ[key] = value

load_env_file(os.path.join(PARALLAX_PATH, ".env"))


@dataclass
class ExecutionResult:
    """Result of a live trade execution."""
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    filled_size: float = 0.0
    avg_price: float = 0.0
    timestamp: Optional[datetime] = None


class LiveExecutor:
    """
    Async adapter to execute RL agent trades on Polymarket.

    Uses py-clob-client directly for order execution.
    Respects KILL_SWITCH and DRY_RUN environment variables.
    """

    def __init__(self, dry_run: bool = True):
        """
        Initialize the live executor.

        Args:
            dry_run: If True, simulate orders without real execution.
                    Can be overridden by DRY_RUN environment variable.
        """
        # Check environment overrides
        env_dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        env_kill_switch = os.getenv("KILL_SWITCH", "on").lower() == "on"

        self.dry_run = dry_run or env_dry_run
        self.kill_switch = env_kill_switch
        self._client = None
        self._client_initialized = False

        # Log configuration
        print(f"  [LiveExecutor] dry_run={self.dry_run}, kill_switch={self.kill_switch}")

    def _init_client(self):
        """Lazy initialization of py-clob-client."""
        if self._client_initialized:
            return

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            api_key = os.getenv("POLYMARKET_API_KEY")
            api_secret = os.getenv("POLYMARKET_API_SECRET")
            api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")

            if not private_key:
                print("  [LiveExecutor] ERROR: POLYMARKET_PRIVATE_KEY not set")
                self._client = None
                self._client_initialized = True
                return

            # Build credentials if available
            creds = None
            if api_key and api_secret and api_passphrase:
                creds = ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase
                )
                print(f"  [LiveExecutor] Using API credentials: {api_key[:20]}...")
            else:
                print("  [LiveExecutor] No API credentials found, deriving new ones...")

            # Get proxy wallet config
            proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS")
            signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))

            # Create client with proxy wallet support
            self._client = ClobClient(
                host="https://clob.polymarket.com",
                key=private_key,
                chain_id=137,  # Polygon
                creds=creds,
                signature_type=signature_type,
                funder=proxy_address
            )

            # Derive credentials if not present
            if not creds:
                try:
                    derived_creds = self._client.derive_api_key()
                    self._client.set_api_creds(derived_creds)
                    print(f"  [LiveExecutor] Derived new API key: {derived_creds.api_key[:20]}...")
                except Exception as e:
                    print(f"  [LiveExecutor] Failed to derive API key: {e}")

            self._client_initialized = True
            print("  [LiveExecutor] py-clob-client initialized")

        except Exception as e:
            print(f"  [LiveExecutor] Failed to initialize client: {e}")
            import traceback
            traceback.print_exc()
            self._client = None
            self._client_initialized = True

    async def execute_trade(
        self,
        condition_id: str,
        token_id: str,
        outcome: str,
        side: str,
        size_usd: float,
        limit_price: float,
    ) -> ExecutionResult:
        """
        Execute a single trade on Polymarket.

        Args:
            condition_id: Market condition ID
            token_id: Token ID for the outcome
            outcome: "Up" or "Down"
            side: "buy" or "sell"
            size_usd: Trade size in USD
            limit_price: Limit price (probability)

        Returns:
            ExecutionResult with success status and order details
        """
        timestamp = datetime.now()

        # Safety check: kill switch
        if self.kill_switch:
            return ExecutionResult(
                success=False,
                error="KILL_SWITCH is ON - trading disabled",
                timestamp=timestamp,
            )

        # Validate inputs
        if size_usd <= 0:
            return ExecutionResult(
                success=False,
                error=f"Invalid size: {size_usd}",
                timestamp=timestamp,
            )

        if not 0 < limit_price < 1:
            return ExecutionResult(
                success=False,
                error=f"Invalid price: {limit_price}",
                timestamp=timestamp,
            )

        # Calculate token size from USD amount
        size_tokens = size_usd / limit_price if limit_price > 0 else 0

        # Dry run: simulate success
        if self.dry_run:
            return ExecutionResult(
                success=True,
                order_id=f"DRY-{condition_id[:16]}-{outcome}",
                filled_size=size_tokens,
                avg_price=limit_price,
                timestamp=timestamp,
            )

        # Live execution using py-clob-client directly
        try:
            self._init_client()

            if self._client is None:
                return ExecutionResult(
                    success=False,
                    error="Polymarket client not available",
                    timestamp=timestamp,
                )

            from py_clob_client.order_builder.constants import BUY, SELL
            from py_clob_client.clob_types import OrderArgs

            # Map side to constant
            order_side = BUY if side.lower() == "buy" else SELL

            # Round price to tick size (0.01)
            rounded_price = round(limit_price, 2)

            # Create and post order using py-clob-client
            print(f"    [CLOB] Placing {side.upper()} order: {size_tokens:.2f} tokens @ {rounded_price}")

            # Build OrderArgs
            order_args = OrderArgs(
                token_id=token_id,
                price=rounded_price,
                size=size_tokens,
                side=order_side,
            )

            # Use create_and_post_order
            order_result = self._client.create_and_post_order(order_args)

            # Extract order ID from result
            order_id = "unknown"
            if isinstance(order_result, dict):
                order_id = order_result.get("orderID") or order_result.get("id") or str(order_result)
            else:
                order_id = str(order_result)

            print(f"    [CLOB] Order placed: {order_id}")

            return ExecutionResult(
                success=True,
                order_id=order_id,
                filled_size=size_tokens,
                avg_price=rounded_price,
                timestamp=timestamp,
            )

        except Exception as e:
            error_msg = str(e)
            # Truncate cloudflare HTML if present
            if "Cloudflare" in error_msg:
                error_msg = "Cloudflare block (403)"
            return ExecutionResult(
                success=False,
                error=f"Order failed: {error_msg}",
                timestamp=timestamp,
            )

    async def get_balance(self) -> Optional[float]:
        """Get current account balance in USD."""
        try:
            self._init_client()
            if self._client:
                # py-clob-client doesn't have a balance method directly
                # Would need to use web3 for this
                return None
        except Exception as e:
            print(f"  [LiveExecutor] Failed to get balance: {e}")
        return None


class SyncLiveExecutor:
    """
    Synchronous wrapper for LiveExecutor.

    The RL agent's execute_action is synchronous, but we need async for some ops.
    This class provides a sync interface by managing an event loop.
    """

    def __init__(self, dry_run: bool = True):
        """
        Initialize the sync executor.

        Args:
            dry_run: If True, simulate orders without real execution.
        """
        self._executor = LiveExecutor(dry_run=dry_run)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get existing event loop or create a new one."""
        try:
            # Try to get running loop (if called from async context)
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            # No running loop - create or reuse our own
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        loop = self._get_or_create_loop()

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            # Not in async context - run directly
            return loop.run_until_complete(coro)

    def execute_trade(
        self,
        condition_id: str,
        token_id: str,
        outcome: str,
        side: str,
        size_usd: float,
        limit_price: float,
    ) -> ExecutionResult:
        """
        Synchronously execute a trade on Polymarket.

        Args:
            condition_id: Market condition ID
            token_id: Token ID for the outcome
            outcome: "Up" or "Down"
            side: "buy" or "sell"
            size_usd: Trade size in USD
            limit_price: Limit price (probability)

        Returns:
            ExecutionResult with success status and order details
        """
        return self._run_async(
            self._executor.execute_trade(
                condition_id=condition_id,
                token_id=token_id,
                outcome=outcome,
                side=side,
                size_usd=size_usd,
                limit_price=limit_price,
            )
        )

    def get_balance(self) -> Optional[float]:
        """Get current account balance in USD."""
        return self._run_async(self._executor.get_balance())

    @property
    def dry_run(self) -> bool:
        """Check if executor is in dry run mode."""
        return self._executor.dry_run

    @property
    def kill_switch(self) -> bool:
        """Check if kill switch is enabled."""
        return self._executor.kill_switch
