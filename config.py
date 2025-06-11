import torch

# env settings
ENV_NAME = "CarRacing-v3"
CONTINUOUS = False
FRAME_STACK = 4
# dqn parameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BUFFER_CAPACITY = 100000
TARGET_UPDATE = 1000
#train parameters
EPISODES = 500
LOG_INTERVAL = 10
SAVE_INTERVAL = 50
#device conv
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#action space
if not CONTINUOUS:
    #actions: [left, right, accelerate, brake, no-op]
    ACTION_SPACE = [
        (-0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.8),
        (0.0, 0.0, 0.0)
    ]
    ACTION_DIM = len(ACTION_SPACE)