import numpy as np
import torch.optim as optim
import gymnasium as gym
import DQN_model as DQN
from config import *
from preprocesing import preprocess_frame, stac_frame
from replay_buffer import ReplayBuffer
from agent import DQNAgent   
import os



def test(model_path, episodes = 10):
    env =    env = gym.make(ENV_NAME, continuous=CONTINUOUS, render_mode="human")
    agent = DQNAgent()
    
    if not os.path.exist(model_path):
        raise FileNotFoundError(f"Model file nod found!")
    
    print(f"Loading model from {model_path}")
    print(f"Model size: {os.path.getsize(model_path)/1024:.2f} KB")
    
    agent.loade_model(model_path)
    
    agent.policy_net.eval()
    agent.epsilon = 0

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)
        state = stac_frame(None, state)
        
        total_reward = 0
        done = False
        
        while not done:
            action_idx = agent.select_action(state)
            action = ACTION_SPACE[action_idx] if not CONTINUOUS else None
            
            next_state, reward, terminated, truncated, _ = env.step(
                action_idx if not CONTINUOUS else action
            )    
            done =  terminated or truncated
            next_state = preprocess_frame(next_state)
            state = stac_frame(state, next_state)
            total_reward += reward
            
        print(f"Test Episode {episode + 1}, Reward: {total_reward:.2f}")
        
if __name__ == "__main__":
   test("models/dqn_car_final.pth")