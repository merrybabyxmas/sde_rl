import asyncio
import json
import time
import random
import torch
# import websockets # Commented out to avoid import error if not installed, but kept for structure

class RealTimeDataCollector:
    def __init__(self, symbol="BTCUSDT", mock=False):
        self.symbol = symbol.lower()
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth10@100ms"
        self.buffer = []
        self.current_latency = 0.005 # Initial value 5ms
        self.mock = mock

    async def connect(self):
        if self.mock:
            await self._run_mock()
        else:
            try:
                import websockets
                async with websockets.connect(self.url) as ws:
                    while True:
                        msg = await ws.recv()
                        self._process_message(msg)
            except ImportError:
                print("websockets library not found. Switching to mock mode.")
                self.mock = True
                await self._run_mock()
            except Exception as e:
                print(f"Connection failed: {e}. Switching to mock mode.")
                self.mock = True
                await self._run_mock()

    async def _run_mock(self):
        while True:
            # Simulate a message
            now = time.time()
            # Mock data structure matching Binance depth update
            data = {
                'E': (now - self.current_latency) * 1000, # Event time
                'b': [[str(50000 + random.random()*100), "1.0"]],
                'a': [[str(50000 + random.random()*100 + 10), "1.0"]]
            }
            # Simulate network delay variability
            await asyncio.sleep(0.1)
            self.current_latency = 0.005 + random.random() * 0.01 # Random latency between 5ms and 15ms

            # Process as if received
            msg = json.dumps(data)
            self._process_message(msg, received_time=time.time())

    def _process_message(self, msg, received_time=None):
        if received_time is None:
            received_time = time.time()

        data = json.loads(msg)

        # Latency calculation: current time - exchange event time
        event_time = data.get('E', received_time * 1000)
        # recv_time is in seconds, event_time is in ms
        self.current_latency = max(0, (received_time - event_time / 1000.0))

        # Process [bid_price, bid_qty, ask_price, ask_qty, latency]
        processed_data = self.preprocess(data)
        self.buffer.append(processed_data)

        if len(self.buffer) > 1000:
            self.buffer.pop(0)

    def preprocess(self, data):
        # Top-1 bid/ask and latency
        try:
            best_bid = float(data['b'][0][0])
            bid_qty = float(data['b'][0][1])
            best_ask = float(data['a'][0][0])
            ask_qty = float(data['a'][0][1])
        except (KeyError, IndexError):
            # Fallback for empty or malformed data
            best_bid = 50000.0
            bid_qty = 1.0
            best_ask = 50010.0
            ask_qty = 1.0

        return torch.tensor([best_bid, bid_qty, best_ask, ask_qty, self.current_latency], dtype=torch.float32)

    def get_latest_data(self):
        if not self.buffer:
            # Return dummy data if buffer empty
            return torch.tensor([50000.0, 1.0, 50010.0, 1.0, 0.005], dtype=torch.float32)
        return self.buffer[-1]
