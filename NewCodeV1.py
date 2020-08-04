import gym
import torch
from torch.nn.utils import spectral_norm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

random_seed = 42

torch.manual_seed(random_seed)

class ActorCriticNet(nn.Module):
    def __init__(self, num_actions, observation_size):
        super().__init__()

        self.afc1 = nn.Linear(observation_size, 32)
        self.afc2 = nn.Linear(32, 32)
        self.afc3 = nn.Linear(32, num_actions)

        self.cfc1 = nn.Linear(observation_size, 32)
        self.cfc2 = nn.Linear(32, 32)
        self.cfc3 = nn.Linear(32, 1)

        self.pfc1 = nn.Linear(observation_size, 32)
        self.pfc2 = nn.Linear(32, 32)
        self.pfc3 = nn.Linear(32, observation_size)

        self.optimizer = optim.SGD(self.parameters(),lr=0.0004)

    def forward(self):
        raise NotImplementedError

    def act(self, x):

        x = F.relu(self.afc1(x))
        x = F.relu(self.afc2(x))
        action_probs = self.afc3(x)

        return action_probs.view(1,-1)

    def eval(self, x):

        x = F.relu(self.cfc1(x))
        x = F.relu(self.cfc2(x))
        pred_value = self.cfc3(x)

        return pred_value.view(1,-1)

    def predict(self, x):

        x = F.relu(self.pfc1(x))
        x = F.relu(self.pfc2(x))
        pred_state = self.pfc3(x)

        return pred_state.view(1, -1)

class FakeNet(nn.Module):
    def __init__(self, observation_size):
        super().__init__()

        self.fc1 = spectral_norm(nn.Linear(16, 32))
        self.fc2 = spectral_norm(nn.Linear(32, 32))
        self.fc3 = spectral_norm(nn.Linear(32, observation_size))

        self.optimizer = optim.SGD(self.parameters(),lr=0.0004)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        fake_state = self.fc3(x)

        return fake_state

class Discriminator(nn.Module):
    def __init__(self, observation_size):
        super().__init__()

        self.fc1 = spectral_norm(nn.Linear(observation_size, 32))
        self.fc2 = spectral_norm(nn.Linear(32, 32))
        self.fc3 = spectral_norm(nn.Linear(32, 1))

        self.optimizer = optim.SGD(self.parameters(),lr=0.0001)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pred_net = self.fc3(x)

        return pred_net


env = gym.make("MountainCar-v0")
env.seed(random_seed)
observation = env.reset()

policy = ActorCriticNet(env.action_space.n, env.observation_space.shape[0])
generator = FakeNet(env.observation_space.shape[0])
discriminator = Discriminator(env.observation_space.shape[0])

P_outer_optimizer = optim.SGD(policy.parameters(), lr=0.00004, momentum=0.79, nesterov=True)
G_outer_optimizer = optim.SGD(generator.parameters(), lr=0.00004, momentum=0.79, nesterov=True)
D_outer_optimizer = optim.SGD(discriminator.parameters(), lr=0.00001, momentum=0.79, nesterov=True)

lossfn = nn.SmoothL1Loss()

update = 200

def reset_inner():

    policy_inner = ActorCriticNet(env.action_space.n, env.observation_space.shape[0])
    policy_inner.load_state_dict(policy.state_dict())

    generator_inner = FakeNet(env.observation_space.shape[0])
    generator_inner.load_state_dict(generator.state_dict())

    discriminator_inner = Discriminator(env.observation_space.shape[0])
    discriminator_inner.load_state_dict(discriminator.state_dict())

    return policy_inner, generator_inner, discriminator_inner

def train(timestep, reward, state, log_prob, pred_value, pred_state, next_pred_value, policy_inner, generator_inner, discriminator_inner):

    # Trains Policy and Discriminator #

    with torch.no_grad():
        fake_state = generator_inner(torch.rand(1,16))

    output = discriminator_inner(state)
    loss_D_real = lossfn(output, torch.ones_like(output))

    output = discriminator_inner(fake_state)
    loss_D_fake = lossfn(output, torch.zeros_like(output))

    loss_D = loss_D_real + loss_D_fake

    predictor_loss = lossfn(pred_state, state)

    reward = -(torch.log(loss_D)).view(1,-1) + torch.tensor(reward).view(1,-1) + torch.log(predictor_loss).view(1,-1)

    advantage = reward + 0.99*next_pred_value - pred_value

    actor_loss = -log_prob * advantage
    critic_loss = advantage**2

    loss_P = actor_loss + critic_loss

    loss_PD = predictor_loss + loss_D + loss_P

    policy_inner.optimizer.zero_grad()
    discriminator_inner.optimizer.zero_grad()
    loss_PD.backward()
    policy_inner.optimizer.step()
    discriminator_inner.optimizer.step()

    # Trains Generator #

    fake_state = generator_inner(torch.rand(1,16))

    output = discriminator_inner(fake_state)
    loss_G = lossfn(output, torch.ones_like(output))

    generator_inner.optimizer.zero_grad()
    loss_G.backward()
    generator_inner.optimizer.step()

    return policy_inner, generator_inner, discriminator_inner

policy_inner, generator_inner, discriminator_inner = reset_inner()

for timestep in range(1000000):

    env.render()

    state = torch.tensor(observation, dtype=torch.float).view(1,-1)

    action_probs = policy_inner.act(state)
    pred_value = policy_inner.eval(state)
    pred_state = policy_inner.predict(state)

    action_softmax = F.softmax(action_probs, dim=1)
    action_distribution = torch.distributions.Categorical(action_softmax)
    action = action_distribution.sample()
    log_prob = action_distribution.log_prob(action)
    action = action.item()

    observation, reward, done, _ = env.step(action)


    state = torch.tensor(observation, dtype=torch.float).view(1,-1)

    next_pred_value = policy_inner.eval(state)

    policy_inner, generator_inner, discriminator_inner = train(timestep,
                                                               reward,
                                                               state,
                                                               log_prob,
                                                               pred_value,
                                                               pred_state,
                                                               next_pred_value,
                                                               policy_inner,
                                                               generator_inner,
                                                               discriminator_inner
                                                              )
    if timestep % update == 0:

        model_list = [[policy, policy_inner], [generator, generator_inner], [discriminator, discriminator_inner]]

        for meta_model, inner_model in model_list:
            for meta, inner in zip(meta_model.parameters(), inner_model.parameters()):
                if meta.grad is None:
                    meta.grad = torch.zeros(meta.size())
                meta.grad.data.add_(meta.data - inner.data)

        policy.load_state_dict(model_list[0][0].state_dict())
        generator.load_state_dict(model_list[1][0].state_dict())
        discriminator.load_state_dict(model_list[2][0].state_dict())

        P_outer_optimizer.step()
        G_outer_optimizer.step()
        D_outer_optimizer.step()

        P_outer_optimizer.zero_grad()
        G_outer_optimizer.zero_grad()
        D_outer_optimizer.zero_grad()

        policy_inner, generator_inner, discriminator_inner = reset_inner()

    print(timestep)

env.close()
