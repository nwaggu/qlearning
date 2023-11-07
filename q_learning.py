import numpy as np
import gridWorld
import random
import matplotlib.pyplot as plt

grid_width = 10
grid_height = 5
alpha = 0.9
actions = 4
action_list = ["up","down","left","right"]

def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v = list(d.values())
     k = list(d.keys())
     return k[v.index(max(v))]

class QLearning():
    def __init__(self, env, alpha, epsilon):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.dict = {}
        self.Q = {}
        self.env.reset()
        #Conversion values
        count = 0
        for x in range(grid_width):
            for y in range(grid_height):
                self.Q[(x,y)] = {}
                if not y+1 > 5 and not (x==7 and y+1<3):
                    self.Q[(x,y)]["up"] = 0
                if not y-1 < 0 and not (x==7 and y-1<3):
                    self.Q[(x,y)]["down"] = 0
                if not x-1 < 0 and not (x-1==7 and y<3):  
                    self.Q[(x,y)]["left"] = 0
                if not x+1 > 9 and not (x+1==7 and y<3):
                    self.Q[(x,y)]["right"] = 0

        
        #Intialize Q Values
        


    def policy(self, current_state):
        random_numb = random.uniform(0,1)
        directions = self.Q[current_state]

        if random_numb < self.epsilon:
            action = keywithmaxval(directions)
        else:
            action = random.choice(list(directions.keys()))
        return action
            

    def updateQ(self, state, action, reward, next_state):
        maxQ = max(self.Q[next_state].values())
        self.Q[state][action] = self.Q[state][action]*(1-self.alpha) + self.alpha * (reward + maxQ)

    def updateSarsa(self, state, action, reward, next_state, next_action):
        self.Q[state][action] = self.Q[state][action]*(1-self.alpha) + self.alpha * (reward + self.Q[next_state][next_action])

    def episode(self,max_steps):
        step = 0
        bonus = 0
        self.env.reset()
        position = self.env.agent
        for step in range(max_steps):
            #Get current location
            state = (position[0],position[1])

            #Get a great action from policy
            action = self.policy(state)

            next_state, reward = self.env.step(action)
            next_state = (next_state[0],next_state[1])
            self.updateQ(state, action, reward, next_state)
            if reward==20:
                bonus+=20
            else:
                bonus-=1
            position = self.env.agent
            #
            step += 1
        return bonus
    
    def episodeSarsa(self,max_steps):
        step = 0
        bonus = 0
        self.env.reset()
        position = self.env.agent
        for step in range(max_steps):
            #Get current location
            state = (position[0],position[1])
            
            #Get a great action from policy
            action = self.policy(state)
            print("current")
            print(state)
            print(action)
            next_state, reward = self.env.step(action)
            print(next_state)
            print(reward)
            if reward==20:
                bonus+=20
            else:
                bonus-=1

            next_state = (next_state[0],next_state[1])
            next_action = self.policy(next_state)
            self.updateSarsa(state, action, reward, next_state, next_action)

            position = self.env.agent
            #
            step += 1
        return bonus


    def train(self,episodes):
        reward_over_time = []
        for _ in range(episodes):
            bonus = self.episode(20)
            reward_over_time.append(bonus)
        return reward_over_time

    def trainSarsa(self,episodes):
        reward_over_time = []
        for _ in range(episodes):
            bonus = self.episodeSarsa(20)
            reward_over_time.append(bonus)
        return reward_over_time

q = QLearning(gridWorld.gridWorld(), alpha, 0.9)
x = list(range(1,100000001))
results_Q = q.train(100000000)
results_Sarsa = q.trainSarsa(100000000)
print(results_Q)
plt.plot(x, results_Q, color='r',label='Q Learning')
plt.plot(x, results_Sarsa, color='b', label='Sarsa')
plt.xlabel("Number of Trials") 
plt.ylabel("Number of Rewards") 
plt.title("Q vs SARSA Learning") 
plt.legend()
plt.show()



class Sarsa():
    def __init__(self) -> None:
        pass





