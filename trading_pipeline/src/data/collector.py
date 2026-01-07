import asyncio
import json
import time
import random
import torch
# import websockets # Kept for structure

class RealTimeDataCollector:
    def __init__(self, symbol="BTCUSDT", mock=False):
        self.symbol = symbol.lower()
        # Depth 20 to get enough data for Top-5
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth20@100ms"
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
            now = time.time()
            # Mock data: 5 levels of bids and asks
            # [Price, Qty]
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
        if received_time is None:
            received_time = time.time()

        data = json.loads(msg)

        event_time = data.get('E', received_time * 1000)
        self.current_latency = max(0, (received_time - event_time / 1000.0))

        processed_data = self.preprocess(data)
        self.buffer.append(processed_data)

        if len(self.buffer) > 2000: # Larger buffer for warm-up
            self.buffer.pop(0)

    def preprocess(self, data):
        # Extract Top-5 Bids and Asks
        # Flattened: [Bid1_P, Bid1_Q, ..., Bid5_P, Bid5_Q, Ask1_P, Ask1_Q, ..., Ask5_P, Ask5_Q, Latency]
        # Total 5*2 + 5*2 + 1 = 21 dimensions

        feature_list = []

        # Bids
        bids = data.get('b', [])
        for i in range(5):
            if i < len(bids):
                feature_list.append(float(bids[i][0])) # Price
                feature_list.append(float(bids[i][1])) # Qty
            else:
                feature_list.append(0.0)
                feature_list.append(0.0)

        # Asks
        asks = data.get('a', [])
        for i in range(5):
            if i < len(asks):
                feature_list.append(float(asks[i][0])) # Price
                feature_list.append(float(asks[i][1])) # Qty
            else:
                feature_list.append(0.0)
                feature_list.append(0.0)

        feature_list.append(self.current_latency)

        return torch.tensor(feature_list, dtype=torch.float32)

    def get_latest_data(self):
        if not self.buffer:
            # Return dummy data if buffer empty (21 dims)
            dummy = [50000.0, 1.0] * 10 + [0.005]
            return torch.tensor(dummy, dtype=torch.float32)
        return self.buffer[-1]

    def get_buffer(self):
        # Return stacked buffer for training
        if not self.buffer:
            return None
        return torch.stack(self.buffer)
