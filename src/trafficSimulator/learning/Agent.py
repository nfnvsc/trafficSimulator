import random
import numpy as np
import pickle
from collections import defaultdict

class Agent:
    def __init__(self, env, signal, epsilon=0.1):
        self.signal = signal
        self.epsilon = epsilon
        self.action_space = [0, 1]
        self.env = env
        #self.q_table = np.zeros([*[500 for _ in range(len(env.state))], len(self.action_space)])
        with open('qtable.pickle', 'rb') as file:
            q_table = pickle.load(file)
            self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)), q_table)
        self.previous_state = None

    @property
    def reward(self):
        def calc_reward(state):
            if state != None:
                return (-sum(state[:-2])) + (state[-1])
            return 0

        reward = calc_reward(self.env.state) - calc_reward(self.previous_state)

        print(reward)
        return reward
    
    def act(self):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = np.argmax(self.q_table[str(self.env.state)]) # Exploit learned values

        print(f"action: {action}")
        self.signal.current_cycle_index = action
        self.previous_state = self.env.state
    
    #At this point the environment already made the step
    def update(self):
        action = self.signal.current_cycle_index
        old_value = self.q_table[str(self.previous_state), action]
        next_max = np.max(self.q_table[str(self.env.state)])
        alpha = 0.6
        gamma = 0.9
        new_value = (1 - alpha) * old_value + alpha * (self.reward + gamma * next_max)
        self.q_table[str(self.previous_state), action] = new_value

        with open('qtable.pickle', 'wb') as file:
            pickle.dump(dict(self.q_table), file)
        

    """
    def update2(self):
        for i in range(1, 100001):
            state = env.reset()
            
            state = [self.n_cars_behind, self.closed_time]

            epochs, penalties, reward, = 0, 0, 0
            done = False
            
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(q_table[state]) # Exploit learned values

                next_state, reward, done, info = env.step(action) 
                
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1
                
            if i % 100 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")
    """
