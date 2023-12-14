import numpy as np
import gridWorld
import random
import matplotlib.pyplot as plt
import seaborn

grid_width = 10
grid_height = 5
alpha = 0.4
actions = 4
action_list = ["up","down","left","right"]

def dictmaxkey(d): 
     v = list(d.values())
     k = list(d.keys())
     return k[v.index(max(v))]

class QLearning():
    def __init__(self, env, alpha, epsilon, discount):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
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
            action = dictmaxkey(directions)
        else:
            action = random.choice(list(directions.keys()))
        return action
            

    def updateQ(self, state, action, reward, next_state):
        maxQ = max(self.Q[next_state].values())
        self.Q[state][action] = self.Q[state][action]*(1-self.alpha) + self.alpha * (reward + self.discount*maxQ)

    def updateSarsa(self, state, action, reward, next_state, next_action):
        self.Q[state][action] = self.Q[state][action]*(1-self.alpha) + self.alpha * (reward + self.discount*self.Q[next_state][next_action])

    def episodeQ(self,max_steps):
        step = 0
        bonus = 0
        self.env.reset()
        position = self.env.agent
        for step in range(max_steps):
            #Get current location
            state = (position[0],position[1])

            #Get a great action from policy
            action = self.policy(state)

            #Get next state and reward
            next_state, reward = self.env.step(action)
            next_state = (next_state[0],next_state[1])
            self.updateQ(state, action, reward, next_state)
            if reward==20:
                bonus+=20
            else:
                bonus-=1
        
            position = self.env.agent
            #
            #self.epsilon=self.epsilon*self.epsilon
            step += 1
        return bonus
    
    def episodeSarsa(self,max_steps):
        step = 0
        bonus = 0
        self.env.reset()
        position = self.env.agent
        state = (position[0],position[1])
        #Get a great action from policy
        action = self.policy(state)
        for step in range(max_steps):
            next_state, reward = self.env.step(action)
            next_state = (next_state[0],next_state[1])
            next_action = self.policy(next_state)
            #Data
            if reward==20:
                bonus+=20
            else:
                bonus-=1
            #Update Q
            self.updateSarsa(state, action, reward, next_state, next_action)
            #Update state
            state = next_state
            action = next_action
            step += 1
            #self.epsilon=self.epsilon*self.epsilon
        return bonus


    def trainQ(self,episodes):
        reward_over_time = []
        for _ in range(episodes):
            bonus = self.episodeQ(20)
            reward_over_time.append(bonus)
        return reward_over_time

    def trainSarsa(self,episodes):
        reward_over_time = []
        for _ in range(episodes):
            bonus = self.episodeSarsa(20)
            reward_over_time.append(bonus)
        return reward_over_time


def createList(output):
    q_values = output.Q
    map = np.zeros((grid_width, grid_height))
    for x in range(grid_width):
        for y in range(grid_height):
            map[x,y] = max(q_values[(x,y)].values())
    map = np.transpose(map)
    map = np.flipud(map)
    s = seaborn.heatmap(map, annot=True, linewidth=0.5)
    xticks = [0,1,2,3,4]
    yticks = [0,1,2,3,4,5,6,7,8,9]
    s.set(xlabel='X Position', ylabel='Y-Position',xticks=yticks,yticks=xticks)
            

q = QLearning(gridWorld.gridWorld(), alpha, 0.9, 0.9)
p = QLearning(gridWorld.gridWorld(), alpha, 0.9, 0.9)
x = list(range(0,200))
results_Q = q.trainQ(200)
results_Sarsa = p.trainSarsa(200)

plt.plot(x, results_Q, color='r',label='Q Learning',linewidth=0.7)
plt.plot(x, results_Sarsa, color='b', label='Sarsa',linewidth=0.7)
plt.title("Q vs SARSA Learning")
plt.ylim(0, 250)
plt.xlabel("Episode #") 
plt.ylabel("Reward Per Episode") 
plt.legend()
plt.show()

plt.title("Q Heatmap") 
plt.xlabel("X Position") 
plt.ylabel("Y Position") 
createList(q)
plt.show()

plt.title("Sarsa Heatmap") 
plt.xlabel("X Position") 
plt.ylabel("Y Position") 
createList(p)
plt.show()





