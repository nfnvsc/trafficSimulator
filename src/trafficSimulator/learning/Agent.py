import random
import numpy as np
import pickle
from collections import defaultdict
import time

class Agent:
    def __init__(self, env, signal, id, epsilon=0.0):
        self.signal = signal
        self.epsilon = epsilon
        self.id = id
        self.action_space = [0, 1]
        self.env = env

        try:
            with open(f'qtable{self.id}.pickle', 'rb') as file:
                q_table = pickle.load(file)
                self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)), q_table)
        except FileNotFoundError:
            self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)))

        self.previous_state = None

        self.timer = time.time()

    @property
    def reward(self):
        def calc_reward(state):
            if state != None:
                if sum(state[:-2]) == 0:
                    return 0
                return (-sum(state[:-2])) + (state[-1])
            return 0

        reward = calc_reward(self.env.state) - calc_reward(self.previous_state)

        return reward
    
    def act(self):
        rdn = random.uniform(0, 1) 

        if rdn < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = np.argmax(self.q_table[str(self.env.state)]) # Exploit learned values

        self.signal.current_cycle_index = action
        self.previous_state = self.env.state
    
    #At this point the environment already made the step
    def update(self):
        action = self.signal.current_cycle_index
        old_value = self.q_table[str(self.previous_state)][action]
        next_max = np.max(self.q_table[str(self.env.state)])
        alpha = 0.6
        gamma = 0.9
        new_value = (1 - alpha) * old_value + alpha * (self.reward + gamma * next_max)
        self.q_table[str(self.previous_state)][action] = new_value

        self.save_qtable()
    
    def save_qtable(self):
        if time.time() - self.timer > 5:
            with open(f'qtable{self.id}.pickle', 'wb') as file:
                pickle.dump(dict(self.q_table), file)
            self.timer = time.time()

    def reset(self):
        self.previous_state = self.env.state
