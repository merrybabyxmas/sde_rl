import asyncio
import json
import time
import random
import torch
import numpy as np

class RealTimeDataCollector:
    def __init__(self, symbol="BTCUSDT", mode="MOCK", lookback=100):
        self.symbol = symbol.lower()
        self.mode = mode
        self.lookback = lookback # For rolling normalization

        # Depth 20 to get enough data for Top-5
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth20@100ms"

        self.buffer = [] # Stores raw processed tensors
        self.price_history = [] # For calculating returns/stats

        self.current_latency = 0.005

        # CCXT Placeholder
        self.exchange = None
        if self.mode == "REAL":
            try:
                import ccxt
                # Placeholder: Need API Keys in env vars or config
                self.exchange = ccxt.binance({
                    'apiKey': 'YOUR_API_KEY',
                    'secret': 'YOUR_SECRET_KEY',
                })
            except ImportError:
                print("CCXT not installed. Reverting to MOCK for execution interface.")
                self.mode = "MOCK"

    async def connect(self):
        if self.mode == "MOCK":
            await self._run_mock()
        else:
            # REAL Mode Connection
            try:
                import websockets
                async with websockets.connect(self.url) as ws:
                    while True:
                        msg = await ws.recv()
                        self._process_message(msg)
            except Exception as e:
                print(f"Connection failed: {e}. Switching to MOCK.")
                self.mode = "MOCK"
                await self._run_mock()

    async def _run_mock(self):
        while True:
            now = time.time()
            base_price = 50000.0 + (random.random() - 0.5) * 100

            bids = [[str(base_price - i*5 - random.random()), str(1.0 + random.random())] for i in range(5)]
            asks = [[str(base_price + 5 + i*5 + random.random()), str(1.0 + random.random())] for i in range(5)]

            data = {
                'E': (now - self.current_latency) * 1000,
                'b': bids,
                'a': asks
            }

            await asyncio.sleep(0.1)
            self.current_latency = max(0.001, 0.005 + (random.random() - 0.5) * 0.002)

            msg = json.dumps(data)
            self._process_message(msg, received_time=time.time())

    def _process_message(self, msg, received_time=None):
        if received_time is None: received_time = time.time()
        data = json.loads(msg)

        event_time = data.get('E', received_time * 1000)
        self.current_latency = max(0, (received_time - event_time / 1000.0))

        # Preprocess Raw Data
        raw_tensor = self.extract_features(data)

        # Normalization
        # We use simple scaling relative to mid-price for levels, or Z-score for rolling stats?
        # A robust way for RL: Input relative distances (%) from mid-price, and log volumes.
        # This is self-normalizing without needing long history for Z-score of price levels.
        # But for 'price' input itself (if needed), we might need returns.
        # Let's use:
        # 1. Mid-Price (Reference)
        # 2. Bids/Asks: (Price - Mid) / Mid * 1000 (Basis points diff)
        # 3. Vols: log(1 + Vol)
        # 4. Latency: Raw or scaled (x100)

        normalized_tensor = self.normalize(raw_tensor)

        self.buffer.append(normalized_tensor)
        if len(self.buffer) > 2000: self.buffer.pop(0)

    def extract_features(self, data):
        # Top-5 Bids/Asks
        feature_list = []

        bids = data.get('b', [])
        asks = data.get('a', [])

        # Helper to safely get float
        def get_level(arr, i):
            if i < len(arr): return float(arr[i][0]), float(arr[i][1])
            return 0.0, 0.0

        # We need Mid Price for normalization reference
        b0_p, _ = get_level(bids, 0)
        a0_p, _ = get_level(asks, 0)
        if b0_p == 0 and a0_p == 0: mid = 50000.0 # Fallback
        elif b0_p == 0: mid = a0_p
        elif a0_p == 0: mid = b0_p
        else: mid = (b0_p + a0_p) / 2.0

        # Store mid price for external usage (reconstruction if needed)
        # We append Mid Price at the end or handle it?
        # The Agent doesn't need absolute price, only relative structure.
        # But 'update_portfolio' needs absolute price.
        # We will return Normalized Tensor, but also keep track of latest absolute price?
        # Actually `get_latest_data` can return a tuple or struct.
        # But existing pipeline expects a tensor.
        # Let's attach Absolute Mid Price as the LAST element (index 21?) or separate method.
        # The prompt asked for normalization.
        # Let's structure the tensor:
        # [Rel_Bid1_P, Log_Bid1_Q, ..., Rel_Ask1_P, Log_Ask1_Q, ..., Latency]
        # And we assume `get_latest_price` is separate.
        # Wait, the pipeline reads `market_data` and extracts price from it.
        # `current_price = (market_data[0] + market_data[10]) / 2.0`.
        # If we normalize, this breaks.
        # We must change `get_latest_data` to return (normalized_state, abs_price, latency).
        # Or, we append abs_price to the tensor for convenience (Agent can ignore it or learn from it).

        # Let's append Abs_Price at end. New Dim = 21 + 1 (Abs Price) ?
        # Or just keep raw and normalize inside Agent?
        # "Normalization Layer" in Collector was requested.
        # So Collector should output Normalized State.
        # We will modify `get_latest_data` to return a dictionary or custom object,
        # OR we pack: [Norm_Feats (20), Latency, Abs_Mid_Price].
        # State Dim = 20 + 1 + 1 = 22.

        # Features
        for i in range(5):
            p, q = get_level(bids, i)
            # Normalize Price: (P - Mid) / Mid * 100 (Percentage)
            p_norm = (p - mid) / mid * 100 if mid > 0 else 0
            # Normalize Qty: Log space
            q_norm = np.log1p(q)
            feature_list.extend([p_norm, q_norm])

        for i in range(5):
            p, q = get_level(asks, i)
            p_norm = (p - mid) / mid * 100 if mid > 0 else 0
            q_norm = np.log1p(q)
            feature_list.extend([p_norm, q_norm])

        feature_list.append(self.current_latency)
        feature_list.append(mid) # Append absolute price for execution logic

        return torch.tensor(feature_list, dtype=torch.float32)

    def normalize(self, raw_tensor):
        # Already normalized in extract_features logic for relative prices
        # Just return it.
        return raw_tensor

    def get_latest_data(self):
        if not self.buffer:
            # Dummy: 22 dims
            return torch.zeros(22, dtype=torch.float32)
        return self.buffer[-1]

    def get_buffer(self):
        if not self.buffer: return None
        return torch.stack(self.buffer)

    # --- Real Exchange Interface (Placeholder) ---
    async def create_order(self, side, quantity, price=None):
        if self.mode == "MOCK":
            return {"status": "filled", "filled": quantity, "price": price}

        if self.exchange:
            # Placeholder for CCXT logic
            # order = await self.exchange.create_order(self.symbol, 'limit', side, quantity, price)
            return {"status": "filled", "filled": quantity, "price": price} # Mock return for now
        return None

    async def fetch_balance(self):
        if self.mode == "MOCK":
            return {"USDT": 1000.0, "BTC": 0.0}
        if self.exchange:
            # bal = await self.exchange.fetch_balance()
            return {"USDT": 1000.0, "BTC": 0.0}
        return None
