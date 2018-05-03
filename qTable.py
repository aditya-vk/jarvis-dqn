import numpy
import gym

# Initialize the environment
env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = numpy.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
learningRate = 0.8
discountFactor = 0.95
numberOfEpisodes = 2000
timeOut = 100

# Container to store returns at the end of every episode
returnList = []

episodeNumber = 0
while episodeNumber <= numberOfEpisodes:

	# Initial State
  state = env.reset()
  cumulativeReward = 0

  for j in range(timeOut):

  	# Choose the best action with some noise. Noise decreases with episode number
    a = numpy.argmax(Q[state,:] + numpy.random.randn(1,env.action_space.n)*(1./(episodeNumber + 1)))

    #Get new state and reward from environment
    newState, reward, isGoal, _ = env.step(a)
    cumulativeReward = cumulativeReward + reward
    
    # Update Q-Table
    Q[state,a] = Q[state,a] + learningRate*(reward + discountFactor*numpy.max(Q[newState,:]) - Q[state,a])
    state = newState
    if isGoal == True:
      break

  # Repeat episode if failed to converge with success
  if isGoal:
  	episodeNumber += 1

  returnList.append(cumulativeReward)

# Print results
print(Q)
print("Score over time: " +  str(sum(returnList)/numberOfEpisodes))
