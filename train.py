import torch
import numpy as np
import torch
import random
import torch.optim as optim
import gymnasium as gym
import DQN_model as DQN
from config import *
from preprocesing import preprocess_frame, stac_frame
from replay_buffer import ReplayBuffer
from agent import DQNAgent   
   
    
def train():
    print(f"Using device {DEVICE}")
    env = gym.make(ENV_NAME,max_episode_steps = 1500, continuous=CONTINUOUS, render_mode = "human")
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
            action_idx = agent.select_action(state_stac)
            action = ACTION_SPACE[action_idx] if not CONTINUOUS else None
            
            next_state, reward, terminated, truncated, _ = env.step(
                    action_idx if not CONTINUOUS else action
                )
            done = terminated or truncated
            
            next_state = preprocess_frame(next_state)
            next_state_stack = stac_frame(state_stac, next_state)
            
            agent.memory.push(state_stac, action_idx, reward, next_state_stack, done)
            
            loss = agent.update()
            
            state_stac = next_state_stack
            total_reword+=reward
            if episode % LOG_INTERVAL == 0:
                env.render()
        
        rewards_history.append(total_reword)
        print(f"Episode { episode + 1}/{EPISODES}, Reward: {total_reword: .2f}, Epsilon: {agent.epsilon: .2f}")

        if((episode + 1) % SAVE_INTERVAL == 0):
            print("bulbulbu")
            save_path = f"models/dqn_episode_{episode+1}.pth"
            agent.save_model(save_path)
            print(f"Model saved to {save_path}")
    
    agent.save_model("models/dqn_final.pth")
    return rewards_history
        
if __name__ == "__main__":
    train()