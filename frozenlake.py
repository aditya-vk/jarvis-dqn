# OpenGym FrozenLake-v0
# -------------------
#
# This code demonstrates use of a basic Q-network (without target network)
# to solve OpenGym FrozenLake-v0 problem.
# 
# Author: Aditya Vamsikrishna
# Date: 26th April 2018.

# Python Headers
import random
import numpy
import math
import IPython

# OpenAI Gym 
import gym

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# ==================================================================
class Environment:
  def __init__(self, envName):
    self.envName = envName
    self.env = gym.make(envName)
    self.maxNumberOfSteps = 100
  
  '''
  Runs one episode of the experiment given the agent
  @param agent Agent that participates in the episode
  '''
  def run(self, agent):
    # Reset the state arbitrarily
    state = self.env.reset()
    cumulativeReward = 0

    # TODO(avk): Change this condition to number of episodes
    for i in range(self.maxNumberOfSteps):
      # Choose action epsilon-greedily
      action = agent.act(state)
      nextState, reward, isDone, information = self.env.step(action)

      agent.train([state, action, reward, nextState])
      state = nextState
      cumulativeReward += reward

      if isDone:
        break

    print("Return: ", cumulativeReward)

# ==================================================================
# Discount Factor
gamma = 0.99

# Greediness
max_epsilon = 1
min_epsilon = 0.01
# Decay of exploration rate
lamda = 0.001

class Agent:
  steps = 0
  epsilon = max_epsilon

  def __init__(self, stateCnt, actionCnt):
    self.stateCnt = stateCnt
    self.actionCnt = actionCnt
    self.model = nn.Linear(self.stateCnt, self.actionCnt)
    self.lossFunction = nn.MSELoss()
    self.opt = optim.RMSprop(self.model.parameters(), lr=0.02)
      
  def act(self, s):
    if random.random() < self.epsilon:
      return random.randint(0, self.actionCnt-1)
    else:
      # TODO (avk): Change this
      return numpy.argmax(self.brain.predictOne(s))
    self.steps += 1
    self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-lamda * self.steps)

  def train(self, sample):

    oldState = sample[0]
    action = sample[1]
    reward = sample[2]
    newState = sample[3]

    torchOldState = torch.from_numpy(numpy.identity(self.stateCnt)[oldState:oldState+1])
    torchOldState = torchOldState.type(torch.FloatTensor)
    torchNewState = torch.from_numpy(numpy.identity(self.stateCnt)[newState:newState+1])
    torchNewState = torchNewState.type(torch.FloatTensor)

    qValues = self.model.forward(torchNewState).data.numpy()
    target = self.model.forward(torchOldState).data.numpy()
    target[0, action] = reward + gamma * numpy.amax(qValues)
    torchTarget = torch.from_numpy(target).type(torch.FloatTensor)

    self.model.zero_grad()
    self.opt.zero_grad()
    loss = self.lossFunction(self.model.forward(torchOldState), torchTarget)
    loss.backward()
    self.opt.step()

# ==================================================================
if __name__ == '__main__':

  PROBLEM = 'FrozenLake-v0'
  env = Environment(PROBLEM)

  stateCnt  = env.env.observation_space.n
  actionCnt = env.env.action_space.n
  agent = Agent(stateCnt, actionCnt)

  numberOfEpisodes = 1

  for episodeNumber in range(numberOfEpisodes):
    env.run(agent)

  IPython.embed()
