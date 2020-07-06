import numpy as np
from collections import deque

import nn as nn

class Agent:
    def __init__(self):
        self.memory = deque([], 100000)
        self.epsilon = 0.999
        self.gamma = 0.85
        self.num_actions = 4

        layer1 = nn.DenseLayer(7, 100, nn.ActivationRELU())
        layer2 = nn.DenseLayer(100, 100, nn.ActivationRELU())
        layer_output = nn.DenseLayer(100, self.num_actions, nn.ActivationOutput())

        self.network = nn.Network([layer1, layer2, layer_output])

    def get_action(self, observation):
            values = self.network.forward(np.array([observation]))
            print(values)
            if (np.random.random() > self.epsilon):
                return np.argmax(values)
            else:
                #print("random")
                return np.random.randint(self.num_actions)

    def remember(self, done, action, observation, prev_obs):
        self.memory.append([done, action, observation, prev_obs])

    def train(self, update_size = 50):
        if len(self.memory) < update_size:
            return
        else:
            batch_indices = np.random.choice(len(self.memory), update_size)
            for index in batch_indices:
                done, action_selected, obs, prev_obs = self.memory[index]

                next_action_values = self.network.forward(np.array([obs]))
                action_values = self.network.forward(np.array([prev_obs]))
                
                experimental_values = np.copy(action_values)

                if done:
                    experimental_values[0][action_selected] = -1
                else:
                    experimental_values[0][action_selected] = 0.5 + self.gamma * np.max(next_action_values[0])

                self.network.backward(action_values, experimental_values, lr = 0.00005)
                self.epsilon = 0 if self.epsilon < 0.01 else self.epsilon*0.998
            
            self.memory = []

