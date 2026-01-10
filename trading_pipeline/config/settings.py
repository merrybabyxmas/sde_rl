import os
import torch

class Config:
    # Mode: MOCK or REAL
    MODE = os.getenv('TRADING_MODE', 'MOCK').upper()

    # Exchange API Keys
    API_KEY = os.getenv('EXCHANGE_API_KEY', '')
    API_SECRET = os.getenv('EXCHANGE_API_SECRET', '')
    SYMBOL = "BTC/USDT"

    # --- Architecture Settings ---
    # RL Agent Hidden Layers
    RL_HIDDEN_DIMS = [256, 128, 64]

    # SDE Network Settings
    LATENT_DIM = 8
    SDE_HIDDEN_DIM = 128 # Width of internal Drift/Diffusion nets

    # --- Device Settings ---
    CUDA_INDEX = int(os.getenv('CUDA_INDEX', '0'))

    # Auto-detect GPU
    if torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{CUDA_INDEX}')
        print(f"CUDA Available. Using GPU: {torch.cuda.get_device_name(CUDA_INDEX)}")
    else:
        DEVICE = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    # --- Hyperparameters ---
    # Training
    SDE_WARMUP_EPOCHS = 5
    RL_LEARNING_RATE = 1e-4
    RL_BATCH_SIZE = 20
    REPLAY_BUFFER_SIZE = 30

    # RL Algo (PPO/PG)
    GAMMA = 0.99
    PPO_EPSILON = 0.2

    # Risk Management
    STOP_LOSS_THRESHOLD = 0.05  # 5%
    COMMISSION_RATE = 0.001     # 0.1%

    # Dimensions (Derived/Fixed)
    STATE_DIM = 21
    PORTFOLIO_DIM = 2
    ACTION_DIM = 1

    # Data
    LOOKBACK_WINDOW = 100


    # config/settings.py 에 추가
    OFFLINE_DATA_SIZE = 30      # 오프라인 학습을 위해 기다릴 최소 데이터 양
    OFFLINE_SDE_EPOCHS = 1      # SDE 오프라인 학습 횟수
    OFFLINE_RL_EPOCHS = 1        # RL 오프라인 학습 횟수


    @classmethod
    def validate(cls):
        if cls.MODE == 'REAL':
            if not cls.API_KEY or not cls.API_SECRET:
                print("WARNING: REAL mode selected but API Keys not found in env vars.")
