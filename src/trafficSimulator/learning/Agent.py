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
    def __init__(self, env, signal, sarsa, config={}):
        self.env = env
        self.signal = signal

        self.action_space = [0, 1]

        self.set_default_config()

        for attr, val in config.items():
            setattr(self, attr, val)

        self.load_qtable()
        self.previous_action = None
        self.previous_state = None
        self.previous_reward = 0

        self.timer = time.time()

        self.using_sarsa = sarsa

    def set_default_config(self):
        self.multithreaded = False
        self.default_val = 0

    def save_qtable(self):
        if self.multithreaded:
            pass

    def load_qtable(self):
        if self.multithreaded:
            self.q_table = DefaultDict(self.env.shared[self.id], self.default_val)


    def _lock(self):
        if self.multithreaded:
            self.lock.acquire()

    def _unlock(self):
        if self.multithreaded:
            self.lock.release()

    @property
    def reward(self):
        def calc_reward(metrics):
            if metrics["collisions"] > 0:
                #print('colision')
                return -100
            if metrics != None:
                return  metrics["avg_speed"]
            return 0

        reward = calc_reward(self.env.metrics)
        #try:
        #    reward = sum([v.x for v in self.signal.roads[0][0].vehicles])/len(self.signal.roads[0][0].vehicles)
        #except ZeroDivisionError:
        #    reward = 5
        #
        #print(reward)
        return reward
    
    def act(self):
        rdn = random.uniform(0, 1) 

        if rdn < self.epsilon:
            action = random.choice(self.action_space)
        else:
            self._lock()
            vals = [self.q_table[str(self.env.state), a] for a in self.action_space]
            action = np.argmax(vals) # Exploit learned values
            self._unlock()
        
        self.previous_state = str(self.env.state)

        self.signal.current_cycle_index = action
    
    # At this point the environment already made the step
    
    def update(self):
        self._lock()

        if not self.using_sarsa:
            action = self.signal.current_cycle_index
            old_value = self.q_table[str(self.previous_state), action]
            vals = [self.q_table[str(self.env.state), a] for a in self.action_space]
            next_max = np.max(vals)
            new_value = (1 - self.alpha) * old_value + self.alpha * (self.reward + self.gamma * next_max)
            #self.previous_reward = self.reward
            self.q_table[str(self.previous_state), action] = new_value

        else:
            # Do sarsa things
            action = self.signal.current_cycle_index
            current_q = self.q_table[str(self.previous_state), action]

            vals = [self.q_table[str(self.env.state), a] for a in self.action_space]
            
            next_action = np.argmax(vals)

            next_q = self.q_table[str(self.env.state), next_action]

            new_q = current_q + self.alpha * (self.reward + self.gamma * next_q - current_q)   # next_max becomes Q(S',A')

            #self.previous_reward = self.reward
            #action = new_action

            self.q_table[str(self.previous_state), action] = new_q

        self._unlock()

    def reset(self):
        self.signal.current_cycle_index = 0
        self.previous_state = None
        #self.previous_action = None
        #if self.using_sarsa:
        #    #self.act()
        #    self.previous_action = self.signal.current_cycle_index
