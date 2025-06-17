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
from car_racing import CarRacing
    
def train():
    print(f"Using device {DEVICE}")

    env = CarRacing(render_mode="human", continuous= CONTINUOUS)
    agent = DQNAgent()
        
    rewards_history = []
    for episode in range(EPISODES):
        step_counter = 0
        current_step = 0
        current_reward_multiplayer = 1
        
        state, _ = env.reset()
        state = preprocess_frame(state)
        state_stac = None
        state_stac = stac_frame(state_stac, state)

        total_reword = 0
        done = False 
        
        while not done:
            action_idx = agent.select_action(state_stac)
            action = ACTION_SPACE[action_idx] if not CONTINUOUS else None
            
            next_state, reward, terminated, truncated, info = env.step(
                    action_idx if not CONTINUOUS else action
                )
            done = terminated or truncated or step_counter == MAX_EPISODE_STEPS
            
            next_state = preprocess_frame(next_state)
            next_state_stack = stac_frame(state_stac, next_state)
            
            #reward modification
            #speed
            if current_step < 7:
                current_step += 1
                if reward > 0:
                    current_reward_multiplayer += 0.15
                    current_step = 0
            else:
                current_reward_multiplayer = 1
                current_step = 0
            
            reward = reward * current_reward_multiplayer
            
            if env.tile_visited_count ==100:
                reward += 100
            elif env.tile_visited_count < 90:
                reward *= 2
            elif env.tile_visited_count < 80:
                reward *= 1.5
                
            
            agent.memory.push(state_stac, action_idx, reward, next_state_stack,terminated)
            
            loss = agent.update()
            step_counter += 1
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