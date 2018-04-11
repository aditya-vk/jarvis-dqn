# #!/usr/bin/python3.5

import numpy
import gym

env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = numpy.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
learningRate = 0.8
discountFactor = 0.95
numberOfEpisodes = 2000
timeOut = 100

returnList = []

for i in range(numberOfEpisodes):

	# Initial State
  state = env.reset()
  cumulativeReward = 0

  for j in range(timeOut):

  	# Choose the best action with some noise. Noise decreases with episode number
    a = numpy.argmax(Q[state,:] + numpy.random.randn(1,env.action_space.n)*(1./(i + 1)))

    #Get new state and reward from environment
    newState, reward, isGoal, _ = env.step(a)
    cumulativeReward = cumulativeReward + reward
    
    # Update Q-Table
    Q[state,a] = Q[state,a] + learningRate*(reward + discountFactor*numpy.max(Q[newState,:]) - Q[state,a])
    state = newState
    if isGoal == True:
      break

  returnList.append(cumulativeReward)

print Q
print "Score over time: " +  str(sum(returnList)/numberOfEpisodes)
