import asyncio
import logging
import random
from datetime import datetime
from dataclasses import dataclass
from alpaca.data.live.crypto import CryptoDataStream

logger = logging.getLogger(__name__)

@dataclass
class MockBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class MockDataStream:
    def __init__(self, config, on_bar_callback):
        self.config = config
        self.on_bar_callback = on_bar_callback
        self.current_price = 68000.0
        self.running = False

    async def start(self):
        logger.info("Starting MOCK Data Stream...")
        self.running = True
        while self.running:
            await asyncio.sleep(getattr(self.config, 'MOCK_SPEED', 2))
            
            # Generate random bar
            change = random.uniform(-100, 100)
            open_price = self.current_price
            close_price = open_price + change
            high_price = max(open_price, close_price) + random.uniform(0, 50)
            low_price = min(open_price, close_price) - random.uniform(0, 50)
            volume = random.uniform(1, 10)
            
            self.current_price = close_price
            
            bar = MockBar(
                timestamp=datetime.now(),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            
            await self.on_bar_callback(bar)

class DataStream:
    def __init__(self, config, on_bar_callback):
        self.config = config
        self.on_bar_callback = on_bar_callback
        
        if getattr(config, 'USE_MOCK', False):
            self.stream_client = MockDataStream(config, on_bar_callback)
        else:
            self.stream_client = CryptoDataStream(config.API_KEY, config.SECRET_KEY)

    async def start(self):
        """Starts the WebSocket stream with robust reconnection logic."""
        if isinstance(self.stream_client, MockDataStream):
            await self.stream_client.start()
            return

        logger.info(f"Subscribing to real-time data for {self.config.SYMBOL}...")
        
        # Subscribe to bars
        self.stream_client.subscribe_bars(self.on_bar_handler, self.config.SYMBOL)

        backoff = 1
        max_backoff = 60

        while True:
            try:
                logger.info("Connecting to Alpaca Stream...")
                await self.stream_client._run_forever()
                # If we exit run_forever cleanly (unlikely), reset backoff
                backoff = 1
            except Exception as e:
                logger.error(f"Stream connection lost: {e}")
                
                # Check for rate limit specific errors
                if "connection limit exceeded" in str(e) or "429" in str(e):
                    logger.warning("Rate limit hit. Increasing backoff.")
                    backoff = max(backoff * 2, 10) # Jump straight to 10s if rate limited
                
                logger.info(f"Reconnecting in {backoff} seconds...")
                await asyncio.sleep(backoff)
                
                # Exponential backoff
                backoff = min(backoff * 2, max_backoff)

    async def on_bar_handler(self, bar):
        """Callback for new bars."""
        # Convert bar to a format suitable for the strategy
        # bar is an instance of alpaca.data.models.Bar
        logger.debug(f"Received bar: {bar}")
        await self.on_bar_callback(bar)
