import math
import random
from collections import namedtuple
from itertools import count

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0
        self.memory = []

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, output):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def get_linear_size(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        linear_h = get_linear_size(get_linear_size(get_linear_size(h)))
        linear_w = get_linear_size(get_linear_size(get_linear_size(w)))
        self.linear = nn.Linear(32 * linear_h * linear_w, output)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.linear(x.view(x.shape[0], -1))
        return x


env = gym.make('CartPole-v0')
env.reset()
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])

hs = []
ws = []


def get_screen():
    screen = env.render(mode='rgb_array')

    # for h in range(len(screen)):
    #     for w in range(len(screen[h])):
    #         if screen[h][w][0] < 255:
    #             hs.append(h)
    #             ws.append(w)

    screen = screen[160:320]
    screen = transform(screen)
    return screen.unsqueeze(0)


def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() < eps_threshold:
        action = torch.tensor([random.randint(0, 1)]).view(1, 1)
    else:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
    return action


def update_model():
    if len(memory) < BATCH_SIZE:
        return
    sample = memory.sample(BATCH_SIZE)
    sample = Transition(*zip(*sample))
    state_batch = torch.cat(sample.state)
    action_batch = torch.cat(sample.action)
    reward_batch = torch.cat(sample.reward)

    next_state_non_mask = torch.tensor(tuple(map(lambda s: s is not None, sample.next_state)))
    next_state_non_values = torch.cat([s for s in sample.next_state if s is not None])
    expected_state_action_values = torch.zeros(BATCH_SIZE, dtype=torch.float)
    expected_state_action_values[next_state_non_mask] = target_net(next_state_non_values).max(1)[0].detach()
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    loss = loss_func(state_action_values, expected_state_action_values.unsqueeze(-1))
    print(f'steps_done: {steps_done}, loss:{loss}')
    loss.backward()
    for param in policy_net.parameters():
        param.grad.clamp_(-1, 1)
    optimizer.step()
    pass


memory = ReplayMemory(1000)
screen = get_screen()
screen_w = screen.shape[-1]
screen_h = screen.shape[-2]
n_action_space = env.action_space.n
policy_net = DQN(screen_h, screen_w, n_action_space)
target_net = DQN(screen_h, screen_w, n_action_space)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.RMSprop(policy_net.parameters())
loss_func = nn.SmoothL1Loss()

BATCH_SIZE = 64
EPISODES = 5
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 200
steps_done = 0


def train():
    for i_episode in range(EPISODES):
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            print(f't:{t}, i_episode:{i_episode}')
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward])

            if done:
                next_state = None
            else:
                last_screen = current_screen
                current_screen = get_screen()
                next_state = current_screen - last_screen

            memory.push(state, action, next_state, reward)
            update_model()
            if done or t > 195:
                break


train()
print(1+1)
