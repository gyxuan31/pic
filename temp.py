import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import random

# --------------------- 加载数据 ---------------------
file = h5py.File('multi_distance.mat', 'r')

multi_distance = np.array(file['multi_distance_true']).transpose(3,2,1,0)  # (6, 55, 15, 3)
multi_prediction = np.array(file['multi_prediction']).transpose(3,2,4,1,0)  # (6, 50, 3, 15, 3)

total_UE = int(file['total_UE'][0][0])
num_RU = int(file['num_RU'][0][0])
num_RB = int(file['num_RB'][0][0])
T = int(file['T'][0][0]) if 'T' in file else 25  # 你自己定
B = float(file['B'][0][0])
P = float(file['P'][0][0])
sigmsqr = float(file['sigmsqr'][0][0])
eta = float(file['eta'][0][0])

file.close()

# --------------------- 系统设置 ---------------------
scenario_idx = 0  # 取第几个场景，可循环
distance = multi_distance[scenario_idx]  # (T, total_UE, num_RU)

rayleigh_gain = np.abs(np.random.randn(total_UE, num_RB))
state_dim = num_RU + num_RB
action_dim = num_RB
hidden_dim = 64

# --------------------- DQN 网络 ---------------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

q_net = DQN()
target_net = DQN()
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epsilon = 0.2
gamma = 0.9
batch_size = 64
buffer = []
max_buffer = 10000
update_freq = 100
episodes = 2000

# --------------------- 计算奖励 ---------------------
def calc_reward(t, user_idx, rb_idx, user_RU, rb_usage):
    signal = P * distance[t, user_idx, user_RU]**(-eta) * rayleigh_gain[user_idx, rb_idx]
    interference = 0
    for other in range(total_UE):
        if other != user_idx and rb_usage[rb_idx] > 0:
            interference += P * distance[t, user_idx, user_RU]**(-eta) * rayleigh_gain[user_idx, rb_idx]
    sinr = signal / (interference + sigmsqr)
    data_rate = B * np.log(1 + sinr)
    return np.log(1 + data_rate)

# --------------------- 经验采样 ---------------------
def sample_batch():
    idx = np.random.choice(len(buffer), batch_size)
    return [buffer[i] for i in idx]

# --------------------- 训练过程 ---------------------
for episode in range(episodes):
    total_reward = 0
    for t in range(T):
        rb_usage = np.zeros(num_RB)
        for u in range(total_UE):
            user_RU = np.argmin(distance[t, u])
            state = np.concatenate([distance[t, u], rb_usage])
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            if random.random() < epsilon:
                action = random.randint(0, num_RB - 1)
            else:
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                action = q_values.argmax().item()

            reward = calc_reward(t, u, action, user_RU, rb_usage)
            next_rb_usage = rb_usage.copy()
            next_rb_usage[action] = 1

            next_state = np.concatenate([distance[t, u], next_rb_usage])
            done = (t == T - 1)

            buffer.append((state, action, reward, next_state, done))
            if len(buffer) > max_buffer:
                buffer.pop(0)

            rb_usage[action] = 1
            total_reward += reward

            if len(buffer) >= batch_size:
                batch = sample_batch()
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.BoolTensor(dones).unsqueeze(1)

                q_pred = q_net(states).gather(1, actions)
                q_next = target_net(next_states).max(1, keepdim=True)[0]
                q_target = rewards + gamma * q_next * (~dones)

                loss = loss_fn(q_pred, q_target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

print("DQN训练完成，直接基于你的.mat数据！")
