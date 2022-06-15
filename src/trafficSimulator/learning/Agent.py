import random
import numpy as np
import pickle
from collections import defaultdict
import time

class DefaultDict:
    def __init__(self, base, default):
        self.base = base
        self.default = default

    def __setitem__(self, key, value):
        self.base[key] = value

    def __getitem__(self, key):
        try:
            return self.base[key]
        except KeyError:
            self.base[key] = self.default
        
        return self.base[key]


class Agent:
    def __init__(self, env, signal, config={}):
        self.env = env
        self.signal = signal

        self.action_space = [0, 1]

        self.set_default_config()

        for attr, val in config.items():
            setattr(self, attr, val)

        self.load_qtable()

        self.previous_state = None

        self.timer = time.time()

    def set_default_config(self):
        self.multithreaded = False
        self.default_val = np.zeros(len(self.action_space))

    def save_qtable(self):
        if self.multithreaded:
            pass
        else:
            if time.time() - self.timer > 5:
                with open(f'qtable{self.id}.pickle', 'wb') as file:
                    pickle.dump(dict(self.q_table), file)
                self.timer = time.time()

    def load_qtable(self):
        if self.multithreaded:
            self.q_table = DefaultDict(self.env.shared[self.id], self.default_val)
        else:
            try:
                with open(f'qtable{self.id}.pickle', 'rb') as file:
                    q_table = pickle.load(file)
                    self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)), q_table)
            except FileNotFoundError:
                self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)))

    def _lock(self):
        if self.multithreaded:
            self.lock.acquire()

    def _unlock(self):
        if self.multithreaded:
            self.lock.release()

    @property
    def reward(self):
        def calc_reward(state):
            if state != None:
                if sum(state[0]) == 0:
                    return 0
                if state[2] > 0:
                    return -100
                return state[1]
            return 0

        reward = calc_reward(self.env.state) - calc_reward(self.previous_state)

        return reward
    
    def act(self):
        rdn = random.uniform(0, 1) 

        if rdn < self.epsilon:
            action = random.choice(self.action_space)
        else:
            self._lock()
            action = np.argmax(self.q_table[str(self.env.state)]) # Exploit learned values
            self._unlock()

        self.signal.current_cycle_index = action
        self.previous_state = self.env.state
    
    #At this point the environment already made the step
    def update(self):
        self._lock()
        action = self.signal.current_cycle_index
        old_value = self.q_table[str(self.previous_state)][action]
        next_max = np.max(self.q_table[str(self.env.state)])
        new_value = (1 - self.alpha) * old_value + self.alpha * (self.reward + self.gamma * next_max)
        self.q_table[str(self.previous_state)][action] = new_value
        self.save_qtable()
        self._unlock()
    
    def reset(self):
        self.previous_state = self.env.state
