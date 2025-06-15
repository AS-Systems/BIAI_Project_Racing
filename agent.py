import torch
import torch.optim as optim
import random
import numpy as np
from DQN_model import DQN
from replay_buffer import ReplayBuffer
from config import *

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
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = LEARNING_RATE)
        
        self.memory = ReplayBuffer(BUFFER_CAPACITY)
        
        self.epsilon = EPSILON_START
        self.steps = 0
    
    def select_action(self, state):
        #epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim -1)
        
        #using DQN to make decision
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def update(self):
        if len(self.memory) <  BATCH_SIZE:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            excepted_q = rewards + (1-dones) * GAMMA * next_q
        
        loss = torch.nn.MSELoss()(current_q.squeeze(), excepted_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.epsilon = max(EPSILON_MIN, self.epsilon* EPSILON_DECAY)
        
        return loss.item()
        
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def loade_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
        
