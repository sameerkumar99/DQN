import gym
import torch
from torch import nn
import numpy as np
from collections import deque
from torch import optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
import sys
import random
import math

# Path for best_model
# class model(nn.Module):
#     def __init__(self, hidden_size):
#         super(model, self).__init__()
#         self.hidden_size = hidden_size
#         self.W1 = nn.Linear(4, 16)
#         self.W2 = nn.Linear(16, 32)
#         self.W3 = nn.Linear(32,24)
#         self.W4 = nn.Linear(24,2)
#         self.relu = nn.ReLU()
    
#     def forward(self, state):
#         x = self.relu(self.W1(state))
#         x = self.relu(self.W2(x))
#         x = self.relu(self.W3(x))
#         x = self.W4(x)
#         return x
# Model for best_model_128
# class model(nn.Module):
#     def __init__(self, hidden_size):
#         super(model, self).__init__()
#         self.hidden_size = hidden_size
#         self.W1 = nn.Linear(4, 16)
#         self.W2 = nn.Linear(16, 128)
#         self.W3 = nn.Linear(128,24)
#         self.W4 = nn.Linear(24,2)
#         self.relu = nn.ReLU()
    
#     def forward(self, state):
#         x = self.relu(self.W1(state))
#         x = self.relu(self.W2(x))
#         x = self.relu(self.W3(x))
#         x = self.W4(x)
#         return x
# Model for 32_64 Works Very Well
# class model(nn.Module):
#     def __init__(self, hidden_size):
#         super(model, self).__init__()
#         self.hidden_size = hidden_size
#         self.W1 = nn.Linear(4, 32)
#         self.W2 = nn.Linear(32, 64)
#         self.W3 = nn.Linear(64,32)
#         self.W4 = nn.Linear(32,2)
#         self.relu = nn.ReLU()
    
#     def forward(self, state):
#         x = self.relu(self.W1(state))
#         x = self.relu(self.W2(x))
#         x = self.relu(self.W3(x))
#         x = self.W4(x)
#         return x

# Model for Path 128_24 Works well for best_model_update_1000_128_24
class model(nn.Module):
    def __init__(self, hidden_size):
        super(model, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(4, 128)
        self.W2 = nn.Linear(128, 24)
        # self.W3 = nn.Linear(64,32)
        self.W3 = nn.Linear(24,2)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.W1(state))
        x = self.relu(self.W2(x))
        # x = self.relu(self.W3(x))
        x = self.W3(x)
        return x

if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"

class DQN:
    def __init__(self, env):
        self.env = env
        self.lr = 1e-4
        self.hidden1 = 128
        self.last_obs = self.env.reset()
        self.eps = 1
        self.target_update_freq = 40
        self.replay_memory = deque()
        self.replay_size = 10000
        self.replay_sz = 0
        self.minibatch = 128
        self.save_rewards = []
        self.eps_decay = 0.99
        self.GAMMA = 0.99
        self.GAM = torch.tensor([self.GAMMA], dtype=torch.float32)
        self.EPISODES = 1500
        self.mean_rewards = []
        self.criterion = nn.MSELoss()
        self.first = False
    
    def q_network(self):
        self.model1 = model(self.hidden1).to(device)
        self.model_target = model(self.hidden1).to(device)
        self.optimizer = optim.Adam(self.model1.parameters(), lr = self.lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[800, 950, 1025, 1075, 1125, 1150], gamma=0.5)

    def get_actions(self, cur_state):
      with torch.no_grad():
        x = self.model1(cur_state)
        return x
    
    def get_targets(self, cur_state):
      with torch.no_grad():
        x = self.model_target(cur_state)
      return x

    def step_env(self):
        cur_state = self.last_obs

        rand_val = random.random()
        with torch.no_grad():
          state = torch.tensor(cur_state)
          state = state.unsqueeze(dim=0).float()
          state = state.to(device)
          q_vals = self.model1(state)[0]
        action = q_vals.argmax().item()
        # print("Action:",action)
        taken = -1
        if rand_val < self.eps:
            if rand_val < self.eps / 2:
                # Take Action 1
                taken = 0
                obs, reward, done, info = self.env.step(0)
            else:
                # Take Action 2
                taken = 1
                obs, reward, done, info = self.env.step(1)
        else:
            # Take Greedy Action
            taken = action
            obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        # print("Taken:",taken)
        # Replay Memory s_(t), a_(t), r_(t), s_(t+1)
        if self.replay_sz > self.replay_size:
            self.replay_memory.popleft()
            self.replay_sz -= 1
        self.replay_memory.append([cur_state, taken, reward, obs, done])
        self.replay_sz += 1
        return obs, reward, done, info
    
    def train(self):
        steps = 0
        for i in range(self.EPISODES):
            self.last_obs = self.env.reset()
            cur_reward = 0
            finish = False
            self.eps = self.eps * self.eps_decay
            while not finish:
                obs, reward, done, info = self.step_env()
                finish = done
                loss = 0
                
                for j in range(self.minibatch):
                    idx = np.random.randint(0, self.replay_sz)
                    st, at, rt, st_1, comp = self.replay_memory[idx]
                    st, st_1 = torch.tensor(st), torch.tensor(st_1)
                    at = torch.tensor(at).to(device)
                    st = st.unsqueeze(dim=0).float()
                    st_1 = st_1.unsqueeze(dim=0).float()
                    st = st.to(device)
                    st_1 = st_1.to(device)
                    q_cur = self.model1(st)
                    with torch.no_grad():
                      q_tar = self.model_target(st_1)
                    Q_sa = torch.sum(nn.functional.one_hot(at, 2)*q_cur, dim=1)
                    best_q_target = torch.max(q_tar, dim=1)[0].detach()
                    if not comp:
                      rt = torch.tensor(rt, device=device)
                      Q_samp = rt + self.GAMMA * best_q_target
                    else:
                      Q_samp = torch.tensor(rt, device=device)
                    loss += self.criterion(Q_sa, Q_samp)
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model1.parameters():
                  param.grad.data.clamp_(-1,1)
                self.optimizer.step()
                
                cur_reward += reward
                steps += 1
                if steps % self.target_update_freq == 0:
                  with torch.no_grad():
                    self.model_target.load_state_dict(self.model1.state_dict())
            self.scheduler.step()
            # print("Cur Reward:{:.4f} Episode : {}".format(cur_reward,i+1))
            self.save_rewards.append(cur_reward)
            self.mean_rewards.append(sum(self.save_rewards[-100:])/100)
            if self.mean_rewards[-1] >= 195.0 and self.first == False:
              self.first = True
              print("Solved Cartpole Bye!!!!!!")
              print("Episode: {}".format(i+1))
              torch.save(self.model1,"best_model")

    def play_policy(self):
        cur_state = self.env.reset()
        cur_reward, finish = 0, False
        steps = 0
        while not finish:
            steps += 1
            self.env.render()
            with torch.no_grad():
                state = torch.tensor(cur_state)
                state = state.unsqueeze(dim=0).float()
                state = state.to(device)
                q_vals = self.model1(state)[0]
            action = q_vals.argmax().item()
            obs, reward, done, info = self.env.step(action)
            cur_state = obs
            cur_reward += reward
            finish = done
        print("Total Reward:",cur_reward)
        print("Total Steps:",steps)
env = gym.make('CartPole-v1')
dqn = DQN(env)
print("Initialized DQN...")
dqn.q_network()
print("Initialized DQN Weights...")
dqn.model1 = torch.load("./best_model_update_1000_128_24")
# dqn.train()
dqn.play_policy()
# rewards = dqn.save_rewards
# x = [i for i in range(dqn.EPISODES)]
# plt.plot(x, rewards)
# plt.show()

            






