import asyncio
import json
import time
import random
import torch
import numpy as np
import os
import traceback
from config.settings import Config

class RealTimeDataCollector:
    def __init__(self, symbol="BTCUSDT", mode=Config.MODE):
        self.symbol_raw = symbol.lower() # for wss
        self.symbol_ccxt = Config.SYMBOL # for ccxt
        self.mode = mode
        self.lookback = Config.LOOKBACK_WINDOW

        # Binance WSS
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol_raw}@depth20@100ms"

        self.buffer = []
        self.current_latency = 0.005

        self.exchange = None
        if self.mode == "REAL":
            try:
                import ccxt.pro as ccxt # Use ccxt.pro for async or standard ccxt with async support
                # If ccxt.pro not available (paid), use ccxt.async_support
            except ImportError:
                import ccxt.async_support as ccxt

            try:
                self.exchange = ccxt.binance({
                    'apiKey': Config.API_KEY,
                    'secret': Config.API_SECRET,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                print(f"Initialized CCXT for {self.mode} mode.")
            except Exception as e:
                print(f"Failed to init CCXT: {e}. Switching to MOCK.")
                self.mode = "MOCK"

    async def connect(self):
        retry_count = 0
        while True:
            try:
                if self.mode == "MOCK":
                    await self._run_mock()
                else:
                    import websockets
                    async with websockets.connect(self.url) as ws:
                        print("Connected to Binance WSS.")
                        retry_count = 0
                        while True:
                            msg = await ws.recv()
                            self._process_message(msg)
            except Exception as e:
                print(f"Connection error: {e}. Retrying in 5s...")
                retry_count += 1
                await asyncio.sleep(5)
                # Fallback to mock if too many failures?
                # if retry_count > 5: self.mode = "MOCK"

    async def _run_mock(self):
        while True:
            try:
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
            except Exception as e:
                print(f"Mock Loop Error: {e}")
                await asyncio.sleep(1)

    def _process_message(self, msg, received_time=None):
        try:
            if received_time is None: received_time = time.time()
            data = json.loads(msg)

            event_time = data.get('E', received_time * 1000)
            self.current_latency = max(0, (received_time - event_time / 1000.0))

            # Robust extraction & normalization
            normalized_tensor = self.extract_features(data)

            # Check validity (NaNs)
            if torch.isnan(normalized_tensor).any():
                # print("Warning: NaN detected in features. Skipping.")
                return

            self.buffer.append(normalized_tensor)
            if len(self.buffer) > Config.REPLAY_BUFFER_SIZE: self.buffer.pop(0)

        except Exception as e:
            # print(f"Msg Processing Error: {e}")
            pass

    def extract_features(self, data):
        feature_list = []
        bids = data.get('b', [])
        asks = data.get('a', [])

        def get_level(arr, i):
            if i < len(arr):
                try:
                    return float(arr[i][0]), float(arr[i][1])
                except (ValueError, IndexError):
                    return 0.0, 0.0
            return 0.0, 0.0

        b0_p, _ = get_level(bids, 0)
        a0_p, _ = get_level(asks, 0)

        if b0_p == 0 and a0_p == 0: mid = 50000.0
        elif b0_p == 0: mid = a0_p
        elif a0_p == 0: mid = b0_p
        else: mid = (b0_p + a0_p) / 2.0

        if mid <= 0: mid = 50000.0 # Safety

        # Features: (Price - Mid)/Mid, Log(Qty)
        for i in range(5):
            p, q = get_level(bids, i)
            p_norm = (p - mid) / mid * 100 if mid > 0 else 0
            q_norm = np.log1p(q)
            feature_list.extend([p_norm, q_norm])

        for i in range(5):
            p, q = get_level(asks, i)
            p_norm = (p - mid) / mid * 100 if mid > 0 else 0
            q_norm = np.log1p(q)
            feature_list.extend([p_norm, q_norm])

        feature_list.append(self.current_latency)
        feature_list.append(mid)

        return torch.tensor(feature_list, dtype=torch.float32)

    def get_latest_data(self):
        if not self.buffer:
            return torch.zeros(22, dtype=torch.float32)
        return self.buffer[-1]

    def get_buffer(self):
        if not self.buffer: return None
        return torch.stack(self.buffer)

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    # --- Real Exchange Interface ---
    async def create_order(self, side, quantity, price=None):
        """
        Executes order via CCXT or returns mock fill.
        """
        if self.mode == "MOCK":
            return {"status": "filled", "filled": quantity, "price": price}

        if self.exchange:
            try:
                # CCXT order structure
                # symbol, type, side, amount, price, params
                if price:
                    order = await self.exchange.create_order(self.symbol_ccxt, 'limit', side, quantity, price)
                else:
                    order = await self.exchange.create_order(self.symbol_ccxt, 'market', side, quantity)

                return order
            except Exception as e:
                print(f"CCXT Create Order Error: {e}")
                return None
        return None

    async def fetch_balance(self):
        """
        Returns {'USDT': float, 'BTC': float}
        """
        if self.mode == "MOCK":
            return {"USDT": 1000.0, "BTC": 0.0}

        if self.exchange:
            try:
                bal = await self.exchange.fetch_balance()
                # Parse standard CCXT balance
                usdt = bal.get('USDT', {}).get('free', 0.0)
                btc = bal.get('BTC', {}).get('free', 0.0)
                return {"USDT": usdt, "BTC": btc}
            except Exception as e:
                print(f"CCXT Fetch Balance Error: {e}")
                return None
        return None
