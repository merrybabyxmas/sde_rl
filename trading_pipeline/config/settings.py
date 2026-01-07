import os

class Config:
    # Mode: MOCK or REAL
    # Controlled via Env Var 'TRADING_MODE', defaults to MOCK
    MODE = os.getenv('TRADING_MODE', 'MOCK').upper()

    # Exchange API Keys
    API_KEY = os.getenv('EXCHANGE_API_KEY', '')
    API_SECRET = os.getenv('EXCHANGE_API_SECRET', '')
    SYMBOL = "BTC/USDT" # CCXT format

    # Hyperparameters
    SDE_WARMUP_EPOCHS = 5
    RL_LEARNING_RATE = 1e-4
    RL_BATCH_SIZE = 20
    REPLAY_BUFFER_SIZE = 2000

    # Risk Management
    STOP_LOSS_THRESHOLD = 0.05  # 5%
    COMMISSION_RATE = 0.001     # 0.1%

    # SDE & Model
    STATE_DIM = 21 # 20 feats + 1 latency
    LATENT_DIM = 8
    PORTFOLIO_DIM = 2
    ACTION_DIM = 1

    # Data
    LOOKBACK_WINDOW = 100

    @classmethod
    def validate(cls):
        if cls.MODE == 'REAL':
            if not cls.API_KEY or not cls.API_SECRET:
                print("WARNING: REAL mode selected but API Keys not found in env vars.")
