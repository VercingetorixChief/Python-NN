import numpy as np
from collections import deque

import nn as nn

class Agent:
    def __init__(self, id):
        self.memory = []
        self.epsilon = 0.999
        self.gamma = 0.85
        self.alpha = 0.7
        self.num_actions = 9
        
        self.prev_obs = [0,0,0, 0,0,0, 0,0,0]
        self.observation = [0,0,0, 0,0,0, 0,0,0]

        self.id = id

        layer1 = nn.DenseLayer(9, 100, nn.ActivationRELU())
        layer2 = nn.DenseLayer(100, 100, nn.ActivationRELU())
        layer_output = nn.DenseLayer(100, self.num_actions, nn.ActivationRELU())

        self.network = nn.Network([layer1, layer2, layer_output])

    def get_actions(self, observation):
            values = self.network.forward(np.array([observation]))
            if (np.random.random() > self.epsilon):
                return values[0]
            else:
                #print("random")
                return np.random.random_integers(0, 8, self.num_actions)

    def remember(self, action, observation):
        self.prev_obs = self.observation
        self.observation = observation
        self.memory.append([action, self.observation, self.prev_obs])

    def train_illegal_move(self, obs, action):
       # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
       # print(obs)
       # print(action)
        action_values = self.network.forward(np.array([obs]))
       # print(action_values)
        experimental_values = np.copy(action_values)
        experimental_values[0][action] = 0.9 * action_values[0][action]
        self.network.backward(action_values, experimental_values, lr = 0.00005)
        action_values = self.network.forward(np.array([obs]))
        #print(action_values)
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def train(self, game_result):
        reward = 0
        if game_result == self.id:
            #won game
            reward = 1
        elif game_result == 3:
            
            #tied
            reward = 0.5
        else:
            reward = 0
        
        first = True
        for index in reversed(range(len(self.memory))):
            action_selected, obs, prev_obs = self.memory[index]
            next_action_values = self.network.forward(np.array([obs]))
            action_values = self.network.forward(np.array([prev_obs]))

            experimental_values = np.copy(action_values)

            if not first:
                experimental_values[0][action_selected] = self.gamma * np.max(next_action_values[0])
            else:
                experimental_values[0][action_selected] = reward
                first = False
            
            self.network.backward(action_values, experimental_values, lr = 0.003)
            self.epsilon = 0 if self.epsilon < 0.01 else self.epsilon*0.998

        self.memory = []
