#############
# First version
#############
import gym
import torch
import torchvision.transforms.functional as TF
from torch.nn.utils import spectral_norm
import random
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(4)
random.seed(13)

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape=[4,84,84], feature_dim=50, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        self.fc = nn.Linear(num_filters * 39 * 39, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        self.outputs = dict()

    def forward_conv(self, obs):
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs):
        h = self.forward_conv(obs)

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = h_norm

        return out


class ActorNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(50, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = self.fc3(x)

        return action_probs

class FakeActorNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(256, 256))
        self.fc2 = spectral_norm(nn.Linear(256, 50))
        self.fc3= spectral_norm(nn.Linear(256, num_actions))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state = self.fc2(x)
        action_probs = self.fc3(x)

        return state, action_probs

class CriticNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(num_actions, 256))
        self.fc2 = spectral_norm(nn.Linear(50, 256))
        self.bfc3 = spectral_norm(nn.Bilinear(256,256,1))

    def forward(self, state, action_probs):
        action_probs = F.relu(self.fc1(action_probs))
        state = F.relu(self.fc2(state))
        x = self.bfc3(state, action_probs)

        return x


env = gym.make("MontezumaRevenge-v0")
observation = env.reset()

policy = ActorNet(env.action_space.n)
fpolicy = FakeActorNet(env.action_space.n)
critic = CriticNet(env.action_space.n)
encQuery = PixelEncoder()
encKey = PixelEncoder()
encKey.load_state_dict(encQuery.state_dict())
P_optimizer = optim.SGD(policy.parameters(), lr=0.0004, momentum=0.79, nesterov=True)
C_optimizer = optim.SGD(critic.parameters(), lr=0.0001, momentum=0.79, nesterov=True)
F_optimizer = optim.SGD(fpolicy.parameters(),lr=0.0004, momentum=0.79, nesterov=True)

lossfn = nn.MSELoss()
crosslossfn = nn.CrossEntropyLoss()
C_accuracies = []
all_obs = []
episode = 1
length = 1

class CURL(nn.Module):
    def __init__(self, feature_dim=50):
        super().__init__()

        self.enc = encQuery
        self.m_enc = encKey

        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))

    def encode(self, x, no_grad=False):

        if no_grad:
            with torch.no_grad():
                out = self.m_enc(x)
        else:
            out = self.enc(x)

        return out

    def compute_logits(self, enc_q, enc_k):

        Wz = torch.matmul(self.W, enc_k.T)
        logits = torch.matmul(enc_q, Wz)
        logits = logits - torch.max(logits, 1)[0][:, None]

        return logits

CURL = CURL()
CPC_optimizer = optim.Adam(CURL.parameters(), lr=1e-3)
Enc_optimizer = optim.Adam(encQuery.parameters(), lr=1e-3)

def train(anchor, positive, timestep):

    Enc_optimizer.zero_grad()
    CPC_optimizer.zero_grad()

    state = CURL.encode(anchor)
    enc_k = CURL.encode(positive, no_grad=True)

    '''Trains Critic'''
    C_optimizer.zero_grad()

    fake_state, fake_action_probs = fpolicy(torch.rand(256))

    action_probs = policy(state)

    output = critic(state, action_probs)
    loss_C_policy = lossfn(output, torch.ones_like(output))
    C_accuracy = int(torch.round(output) == torch.ones_like(output))

    output = critic(fake_state, fake_action_probs)
    loss_C_fake = lossfn(output, torch.ones_like(output))
    C_accuracy += int(torch.round(output) == torch.zeros_like(output))

    loss_C = loss_C_policy + loss_C_fake

    C_accuracy /= 2

    loss_C.backward(retain_graph=True)
    C_optimizer.step()

    '''Trains Policy'''
    P_optimizer.zero_grad()

    action_probs = policy(state)

    output = critic(state, action_probs)
    loss_P = lossfn(output, torch.ones_like(output))

    P_train_loss = loss_P.item()

    loss_P.backward(retain_graph=True)
    P_optimizer.step()


    '''Trains Fake Policy'''
    F_optimizer.zero_grad()

    fake_state, fake_action_probs = fpolicy(torch.rand(256))

    output = critic(fake_state, fake_action_probs)
    loss_F = lossfn(output, torch.ones_like(output))

    F_train_loss = loss_F.item()

    loss_F.backward()
    F_optimizer.step()

    logits = CURL.compute_logits(state, enc_k)
    labels = torch.arange(logits.shape[0]).long()
    crossloss = crosslossfn(logits, labels)

    crossloss.backward()

    Enc_optimizer.step()
    CPC_optimizer.step()

    if timestep % 2 == 0:
        for param, target_param in zip(encQuery.parameters(), encKey.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

    return action_probs.detach(), P_train_loss, F_train_loss, C_accuracy

def preprocessing(image):
    image_data = cv2.cvtColor(cv2.resize(image, (100, 100)), cv2.COLOR_RGB2GRAY)
    image_data = np.reshape(image_data,(100, 100, 1))
    image_tensor = image_data.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor

for time in range(1000000):

    P_train_loss = "Not yet"
    F_train_loss = "Not yet"
    action_probs = "Still random"
    C_accuracy = 0

    if time >= 4:

        state = torch.cat(all_obs, dim=0)

        state = TF.to_pil_image(state)

        anchor = TF.crop(state, random.randint(0,15), random.randint(0,15), 84, 84)
        positive = TF.crop(state, random.randint(0,15), random.randint(0,15), 84, 84)

        anchor = TF.to_tensor(anchor).unsqueeze(0)
        positive = TF.to_tensor(positive).unsqueeze(0)

        action_probs, P_train_loss, F_train_loss, C_accuracy = train(anchor, positive, time)

        all_obs.pop(0)

        action_probs = F.softmax(action_probs, dim=1)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        action = action.item()
    else:
        action = env.action_space.sample()

    if time >= 100:

        C_accuracies.pop(0)


    observation, _, done, _ = env.step(action)
    length += 1

    if done:
        episode += 1
        length = 1
        observation = env.reset()

    C_accuracies.append(C_accuracy)
    obs = preprocessing(observation)
    all_obs.append(obs)

    print(f"Episode/Time:{episode}-{length}|P_train_loss:{P_train_loss}|F_train_loss:{F_train_loss}|C_accuracy:{sum(C_accuracies)/len(C_accuracies)}|Action:{action}")
env.close()
