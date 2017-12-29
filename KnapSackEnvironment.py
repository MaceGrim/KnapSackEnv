# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:38:47 2017

@author: Mason
"""

import random as rd
import pandas as pd
import numpy as np
from gym import spaces
import gym

class KnapSackEnvironment(object):
    
    def __init__(self):
        # How many items: Human Input Needed
        self.item_count = 5
        
        # Create Items
        self.utilities = [rd.triangular(low=5, high=20) for i in range(self.item_count)]
        self.costs = [rd.triangular(low=5, high=20) for i in range(self.item_count)]
        self.util_costs_pairs = [(x, y) for x, y in zip(self.utilities, self.costs)]
        self.state = self.util_costs_pairs
        
        # Cost Constraint: Human Input Needed
        self.cost_limit = 12.5*3
        
        self.action_space = spaces.MultiBinary(self.item_count)
        self.observation_space = spaces.Box(low=5, high=20, shape=(self.item_count, 2))

    def step(self, action):
        chosen_pairs = []
        
        indexes = 0
        for decision in action:
            if decision:
                chosen_pairs.append(self.util_costs_pairs[indexes])
            indexes += 1
        
        total_cost = sum([item[1] for item in chosen_pairs])
        
        if total_cost > self.cost_limit:
            reward = -30
        else:
            reward = sum([item[0] for item in chosen_pairs])
            
        done = True
            
        return np.array(self.state), reward, done, {}
        
    def reset(self):
        self.utilities = [rd.triangular(low=5, high=20) for i in range(self.item_count)]
        self.costs = [rd.triangular(low=5, high=20) for i in range(self.item_count)]
        self.util_costs_pairs = [(x, y) for x, y in zip(self.utilities, self.costs)]
        return self.state
    
    def get_state(self):
        return self.state
    
env = KnapSackEnvironment()
print(env.step([1, 1, 0, 0, 0]))