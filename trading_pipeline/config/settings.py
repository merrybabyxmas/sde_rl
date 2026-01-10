import os
import torch

class Config:
    # Mode: MOCK or REAL
    MODE = os.getenv('TRADING_MODE', 'MOCK').upper()

    # Exchange API Keys
    API_KEY = os.getenv('EXCHANGE_API_KEY', '')
    API_SECRET = os.getenv('EXCHANGE_API_SECRET', '')
    SYMBOL = "BTC/USDT"

    # --- RL Algorithm Selection ---
    # Options: 'PPO', 'SAC'
    RL_ALGO = os.getenv('RL_ALGO', 'SAC').upper()

    # --- Architecture Settings ---
    RL_HIDDEN_DIMS = [256, 256] # Deeper for SAC usually better
    LATENT_DIM = 8
    SDE_HIDDEN_DIM = 64

    # --- Device Settings ---
    CUDA_INDEX = int(os.getenv('CUDA_INDEX', '0'))

    if torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{CUDA_INDEX}')
        print(f"CUDA Available. Using GPU: {torch.cuda.get_device_name(CUDA_INDEX)}")
    else:
        DEVICE = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    # --- Hyperparameters ---
    # Common
    SDE_WARMUP_EPOCHS = 5
    RL_LEARNING_RATE = 3e-4 # Standard for SAC/PPO
    RL_BATCH_SIZE = 256 # Larger batch for SAC
    REPLAY_BUFFER_SIZE = 100000 # SAC needs large buffer
    GAMMA = 0.99

    # PPO Specific
    PPO_EPSILON = 0.2
    PPO_K_EPOCHS = 10
    PPO_ENTROPY_COEF = 0.01

    # SAC Specific
    SAC_TAU = 0.005
    SAC_ALPHA = 0.2
    SAC_AUTO_ENTROPY_TUNING = True

    # Risk Management
    STOP_LOSS_THRESHOLD = 0.05
    COMMISSION_RATE = 0.001

    # Dimensions
    STATE_DIM = 21
    PORTFOLIO_DIM = 2
    ACTION_DIM = 1

    # Data
    LOOKBACK_WINDOW = 100

    @classmethod
    def validate(cls):
        if cls.MODE == 'REAL':
            if not cls.API_KEY or not cls.API_SECRET:
                print("WARNING: REAL mode selected but API Keys not found.")
