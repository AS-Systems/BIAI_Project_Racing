import torch
import random
import torch.optim as optim
import gymnasium as gym
import DQN_model as DQN
from config import *
from preprocesing import preprocess_frame, stac_frame
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self):
        self.action_dim = ACTION_DIM
        self.device = DEVICE

        #Main network that is learning
        self.policy_net = DQN(ACTION_DIM).to(self.device) 
        #copy of main network for stabilising learning
        self.target_net = DQN(ACTION_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam()
        
        self.memory = ReplayBuffer(BUFFER_CAPACITY)
        
        self.epsilon = EPSILON_START
        self.steps = 0
    
    def select_action(self, state):
        #epsilon-greedy
        if random.randint() < self.epsilon:
            return random.randint(0, self.action_dim -1)
        
        #using DQN to make decision
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
        

    
def train():
    env = gym.make(ENV_NAME, continuous=CONTINUOUS, render_mode = "human")
    agent = DQNAgent()
        
    rewards_history = []
    for episode in range(EPISODES):
        
        state, _ = env.reset()
        state = preprocess_frame(state)
        state_stac = None
        state_stac = stac_frame(state_stac, state)

        total_reword = 0
        done = False 
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(
                    action
                )
            if terminated:
                done = True
                
        env.close()
        
if __name__ == "__main__":
    train()