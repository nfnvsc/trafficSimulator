import random
import numpy as np

class Agent:
    def __init__(self, env, signal, epsilon=0.1):
        self.signal = signal
        self.epsilon = epsilon
        self.action_space = [0, 1]
        self.env = env
        self.q_table = np.zeros([*[500 for _ in range(len(env.state))], len(self.action_space)])
        self.previous_state = None

    def reward(self):
        return self.env.state[-1]
    
    def act(self):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = np.argmax(self.q_table[self.env.state]) # Exploit learned values
            print(self.q_table[self.env.state])

        #print(f"action: {action}")
        self.signal.current_cycle_index = action
        self.previous_state = self.env.state
    
    #At this point the environment already made the step
    def update(self):
        action = self.signal.current_cycle_index
        old_value = self.q_table[self.previous_state, action]
        next_max = np.max(self.q_table[self.env.state])
        alpha = 0.1
        gamma = 0.9
        new_value = (1 - alpha) * old_value + alpha * (self.reward() + gamma * next_max)
        self.q_table[self.previous_state, action] = new_value

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
