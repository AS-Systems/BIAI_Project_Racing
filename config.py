import torch

ENV_NAME = "CarRacing-v3"
FRAME_STACK = 4
CONTINUOUS = False
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
BUFFER_CAPACITY = 100000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPISODES = 200
EPSILON_START = 1.0
GAMMA = 0.99

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