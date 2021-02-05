import math
import random
from itertools import count
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from collections import namedtuple
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

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
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def get_linear_input_size(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        linear_w = get_linear_input_size(get_linear_input_size(get_linear_input_size(w)))
        linear_h = get_linear_input_size(get_linear_input_size(get_linear_input_size(h)))
        self.linear = nn.Linear(32 * linear_h * linear_w, outputs)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


env = gym.make('CartPole-v0')
env.reset()

transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])


def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen)
    screen = torch.from_numpy(screen)
    return transform(screen).unsqueeze(0)


# plt.ion()
# plt.figure()
# plt.imshow((get_screen()).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
# plt.title('example extracted screen')
# plt.show()

memory = ReplayMemory(10000)

screen = get_screen()
screen_height = screen.shape[2]
screen_width = screen.shape[3]
n_action_space = env.action_space.n
policy_net = DQN(screen_height, screen_width, n_action_space)
target_net = DQN(screen_height, screen_width, n_action_space)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.RMSprop(policy_net.parameters())
loss_func = nn.SmoothL1Loss()

EPISODES = 50
UPDATE_EPISODES = 10
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 200
steps_done = 0

episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('training...')
    plt.xlabel('episode')
    plt.ylabel('durations')
    plt.plot(durations_t.numpy())




def select_action(state):
    global steps_done
    eps_threshold = EPS_START + (EPS_END - EPS_START) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() < eps_threshold:
        action = torch.tensor(random.randint(0, 1)).view(1, 1)
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
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_not_none_mask = torch.tensor(tuple(map(lambda s: s is not None, sample.next_state)))
    next_state_not_none_values = torch.cat([s for s in sample.next_state if s is not None])
    next_state_values = torch.zeros(BATCH_SIZE, dtype=torch.float)
    next_state_values[next_state_not_none_mask] = target_net(next_state_not_none_values).max(1)[0].detach()
    expected_state_action_values = next_state_values * GAMMA + reward_batch
    loss_func(state_action_values, expected_state_action_values.unsqueeze(1))


for episode in range(EPISODES):
    print(f'episode:{episode}')
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for i in count():
        print(f'i:{i}')
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
        state = next_state
        update_model()
        if done or i > 195:
            episode_durations.append(i + 1)
            plot_durations()
            break
    if episode % UPDATE_EPISODES == 0:
        target_net.load_state_dict(policy_net.state_dict())
print('Complete!')
