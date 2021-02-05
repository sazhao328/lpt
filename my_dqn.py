import math
import random
from collections import namedtuple
from itertools import count

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('CartPole-v0').unwrapped
env.reset()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = torch.from_numpy(screen.copy())
    transformer = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])
    screen = transformer(screen).unsqueeze(0)

    return screen


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def get_conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        out_h = get_conv2d_size_out(get_conv2d_size_out(get_conv2d_size_out(h)))
        out_w = get_conv2d_size_out(get_conv2d_size_out(get_conv2d_size_out(w)))
        linear_input_size = out_h * out_w * 32
        self.fc = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x.view(x.size(0), -1))
        return x


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

screen = get_screen()
_, channels, screen_height, screen_width = screen.shape
n_actions = env.action_space.n
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.05)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    action_random = random.random()
    eps_threshold = EPS_START + (EPS_END - EPS_START) * math.exp(-1 * steps_done / EPS_DECAY)
    if action_random < eps_threshold:
        return torch.tensor([[random.randrange(n_actions)]], device=device)
    else:
        with torch.no_grad():
            pred = policy_net(state).max(1)[1].view(1, 1)
            return pred


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    sample = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*sample))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_batch_not_none_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                                  dtype=torch.bool)
    next_state_batch_not_none_values = torch.cat(
        [next_state for next_state in batch.next_state if next_state is not None])
    next_state_batch = torch.zeros(BATCH_SIZE, dtype=torch.float)
    next_state_batch[next_state_batch_not_none_mask] = target_net(next_state_batch_not_none_values).max(1)[0].detach()
    expected_state_action_values = next_state_batch * GAMMA + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50


def train():
    for i in range(num_episodes):
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = last_screen - current_screen

        for t in count():
            action = select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward])
            if done:
                next_state = None
            else:
                last_screen = current_screen
                current_screen = get_screen()
                next_state = current_screen - last_screen
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
            if done:
                break
        if i % TARGET_UPDATE == 0:
            print('update target net')
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete!')


train()
