import os, time, gym
import numpy as np
from tqdm import tqdm

#Environment
env = gym.make('FrozenLake-v0')
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

#Q-table
Q = np.zeros((state_space_size, action_space_size))

#Hyper-parameters
lr = 0.1
gamma = 0.99
epsilon = 1
min_epsilon = 0.01
max_epsilon = 1
epsilon_decay_rate = 0.001
episodes = 10000
max_steps_per_episode = 400


#Get available actions
def get_available_action(state):
    actions_Q_values = Q[state, :]
    return actions_Q_values


#Select action
def select_action(actions, epsilon=epsilon):
    rand_explorative_treshold = np.random.uniform(0, 1)

    #exploitation approach
    if (rand_explorative_treshold > epsilon):
        action = np.argmax(action)
    
    #exploitative approach
    else:
        action = env.action_space.sample()

    return action


#Update Q-table
def updata(state, action, lr=lr, gamma=gamma):
    new_state, reward, done, info = env.step(action)
    Q[state, action] = Q[state, action] + lr*(reward + gamma*( np.max(Q[new_state, :]) ) - Q[state, action] )

    return (reward, new_state, done)



#training process
for episode in tqdm(range(episodes)):
    state = env.reset()
    done = False

    for step in range(max_steps_per_episode):
        actions = get_available_action(state)
        action = select_action(actions)
        reward, new_state, done = updata(state, action)
        state = new_state

        if (done == True):
            break

    #epsilon_decay
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate*episode)
    

#inference
for episode in range(10):
    state = env.reset()
    done = False

    for step in range(max_steps_per_episode):
        os.system('cls')
        env.render()
        actions = get_available_action(state)
        action = np.argmax(actions)
        new_state, reward, done, info = env.step(action)

        if (done == True):
            os.system('cls')
            env.render()
            if (reward == 1):
                print("Goal has been reached")
                time.sleep(3)
            else:
                print("Failed to reach goal")
                time.sleep(3)
            break
        state = new_state



