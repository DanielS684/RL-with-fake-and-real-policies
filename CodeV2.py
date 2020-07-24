
#############
# Second version
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
        self.fc3 = spectral_norm(nn.Linear(256, num_actions))
        self.fc4 = spectral_norm(nn.Linear(256, 50))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        pre_state = self.fc2(x)
        action_probs = self.fc3(x)
        cur_state = self.fc4(x)

        return pre_state, action_probs, cur_state

class CriticNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(num_actions, 256))
        self.fc2 = spectral_norm(nn.Linear(50, 256))
        self.fc3 = spectral_norm(nn.Linear(50, 256))
        self.bfc4 = spectral_norm(nn.Bilinear(256,256,256))
        self.bfc5 = spectral_norm(nn.Bilinear(256,256,1))

    def forward(self, pre_state, action_probs, cur_state):
        action_probs = F.relu(self.fc1(action_probs))
        pre_state = F.relu(self.fc2(pre_state))
        cur_state = F.relu(self.fc3(cur_state))
        x = F.relu(self.bfc4(pre_state, action_probs))
        x = self.bfc5(x, cur_state)

        return x


env = gym.make("MontezumaRevenge-v4")
observation = env.reset()

policy = ActorNet(env.action_space.n).to("cuda:0")
fpolicy = FakeActorNet(env.action_space.n).to("cuda:0")
critic = CriticNet(env.action_space.n).to("cuda:0")
encQuery = PixelEncoder().to("cuda:0")
encKey = PixelEncoder().to("cuda:0")
encKey.load_state_dict(encQuery.state_dict())
P_optimizer = optim.SGD(policy.parameters(), lr=0.0004, momentum=0.79, nesterov=True)
C_optimizer = optim.SGD(critic.parameters(), lr=0.0001, momentum=0.79, nesterov=True)
F_optimizer = optim.SGD(fpolicy.parameters(),lr=0.0004, momentum=0.79, nesterov=True)

lossfn = nn.SmoothL1Loss()
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

    def rand_aug(self, state):

        state = TF.to_pil_image(state)

        anchor = TF.crop(state, random.randint(0,15), random.randint(0,15), 84, 84)
        positive = TF.crop(state, random.randint(0,15), random.randint(0,15), 84, 84)

        anchor = TF.to_tensor(anchor).unsqueeze(0)
        positive = TF.to_tensor(positive).unsqueeze(0)

        return anchor, positive


CURL = CURL().to("cuda:0")
CPC_optimizer = optim.Adam(CURL.parameters(), lr=1e-3)
Enc_optimizer = optim.Adam(encQuery.parameters(), lr=1e-3)

def train(pre_state, pre_enc_k, cur_state, action_probs, timestep):

    cur_anchor, cur_positive = CURL.rand_aug(cur_state)

    cur_state = CURL.encode(cur_anchor.to("cuda:0"))
    cur_enc_k = CURL.encode(cur_positive.to("cuda:0"), no_grad=True)

    '''Trains Critic'''
    C_optimizer.zero_grad()

    fake_pre_state, fake_action_probs, fake_cur_state = fpolicy(torch.rand(1,256).to("cuda:0"))

    output = critic(pre_state, action_probs, cur_state)
    loss_C_policy = lossfn(output, torch.ones_like(output))
    C_accuracy = int(torch.round(output) == torch.ones_like(output))

    output = critic(fake_pre_state, fake_action_probs, fake_cur_state)
    loss_C_fake = lossfn(output, torch.ones_like(output))
    C_accuracy += int(torch.round(output) == torch.zeros_like(output))

    loss_C = loss_C_policy + loss_C_fake

    C_accuracy /= 2

    loss_C.backward(retain_graph=True)
    C_optimizer.step()

    '''Trains Policy'''
    P_optimizer.zero_grad()

    output = critic(pre_state, action_probs, cur_state)
    loss_P = lossfn(output, torch.ones_like(output))

    P_train_loss = loss_P.item()

    loss_P.backward(retain_graph=True)
    P_optimizer.step()


    '''Trains Fake Policy'''
    F_optimizer.zero_grad()

    fake_pre_state, fake_action_probs, fake_cur_state = fpolicy(torch.rand(1,256).to("cuda:0"))

    output = critic(fake_pre_state, fake_action_probs, fake_cur_state)
    loss_F = lossfn(output, torch.ones_like(output))

    F_train_loss = loss_F.item()

    loss_F.backward()
    F_optimizer.step()

    logits = CURL.compute_logits(pre_state, pre_enc_k)
    labels = torch.arange(logits.shape[0]).long().to("cuda:0")
    pre_crossloss = crosslossfn(logits, labels)

    logits = CURL.compute_logits(cur_state, cur_enc_k)
    labels = torch.arange(logits.shape[0]).long().to("cuda:0")
    cur_crossloss = crosslossfn(logits, labels)

    crossloss = pre_crossloss + cur_crossloss

    Enc_optimizer.zero_grad()
    CPC_optimizer.zero_grad()

    crossloss.backward()

    Enc_optimizer.step()
    CPC_optimizer.step()

    if timestep % 2 == 0:
        for param, target_param in zip(encQuery.parameters(), encKey.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

    return P_train_loss, F_train_loss, C_accuracy

def preprocessing(image):

    image_data = cv2.cvtColor(cv2.resize(image, (100, 100)), cv2.COLOR_RGB2GRAY)
    image_data = np.reshape(image_data,(100, 100, 1))
    image_tensor = image_data.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)

    return image_tensor

for timestep in range(1000000):

    P_train_loss = "Not yet"
    F_train_loss = "Not yet"
    action_probs = "Still random"
    C_accuracy = 0

    if timestep >= 4:

        pre_state = torch.cat(all_obs, dim=0)

        pre_anchor, pre_positive = CURL.rand_aug(pre_state)

        pre_state = CURL.encode(pre_anchor.to("cuda:0"))
        pre_enc_k = CURL.encode(pre_positive.to("cuda:0"), no_grad=True)

        action_probs = policy(pre_state)

        all_obs.pop(0)

        action_softmax = F.softmax(action_probs, dim=1)
        action_distribution = torch.distributions.Categorical(action_softmax)
        action = action_distribution.sample().to("cpu")
        action = action.item()

    else:
        action = env.action_space.sample()

    if timestep >= 100:

        C_accuracies.pop(0)


    observation, _, done, _ = env.step(action)
    length += 1

    if done:
        episode += 1
        length = 1
        observation = env.reset()

    obs = preprocessing(observation)
    all_obs.append(obs)

    if timestep >= 4:

        cur_state = torch.cat(all_obs, dim=0)

        P_train_loss, F_train_loss, C_accuracy = train(pre_state, pre_enc_k, cur_state, action_probs, timestep)

    C_accuracies.append(C_accuracy)

    print(f"Episode/Time:{episode}-{length}|P_train_loss:{P_train_loss}|F_train_loss:{F_train_loss}|C_accuracy:{sum(C_accuracies)/len(C_accuracies)}|Action:{action_softmax}")
    
env.close()
