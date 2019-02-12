
## (Unofficial) Udacity's How to Train a Quadcopter Best Practices


### Useful links 
- https://knowledge.udacity.com/questions/3128
    - [Mirror link](https://github.com/WittmannF/quadcopter-best-practices/blob/master/knowledge_udacity_question_3128.pdf)
- https://discussions.udacity.com/t/anyone-got-quadcopter-v2-to-fly/655471
- https://www.youtube.com/watch?v=0R3PnJEisqk
- https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
- https://arxiv.org/pdf/1509.02971.pdf

### How to get started
Since the DDPG algorithm is already provided, your main goal is to define the reward function to make the agent learn your choice of task. Ideally this reward function should be continuous and differentiable.

### Main Tips
- Ideally the reward function should be normalized between -1 and 1 (except for colisions) in order to the NN better learn the gradients. The method `np.clip` can be used for this purpose.
Check the learning rate parameter in `Adam(lr=...)`. Lower learning rates might lead to better learning results.
- In order to debug the agent, after training it is highly advisable to visualize it. Check the Visualization section. 
- Also try to visualize the reward function as a heatmap in order to better debug it. Check the visualization section as well. 
- Keep in mind that the z = 0 is considered the floor. 
- Don’t initialize the agent on z=0 since it can be too unstable to fly and easily crash.
- When flying, it is important to avoid crashes by penalizing colisions to the floor. The penalization has to be very high in order to compensate the accumulated positive reward. You don’t need to keep it between -1 and 1. Here’s an example on how this can be done:
```
# Check if done is true before the runtime finished
if self.sim.done and self.sim.runtime > self.sim.time:
    reward = -... # 
```

- You can choose one of 4 tasks: takeoff, hovering, landing and (forgot the last one). Usually the easiest task to get started is the takeoff. 
    - For the takeoff task, put the z-axis in a very high distance, like z=300. 
    - For the hovering task keep in mind that 1/rpm can make a huge difference between the agent going up or down. Try lowering the minimum and maximum speed range. 
    - For the landing taks, try to include the speed in the reward function. Ideally you should reward very low speed when the agent is close to the origin. 
    - If you want the agent to fly from point A to point B, it is important to make the agent aknowledge when it reached the destination and finalize the episode when this happens. Please check the first answer of [this reference](https://knowledge.udacity.com/questions/3128) for more details. 
- It might be too unstable for the agent to learn to control the exact position of x, y and z with only 1000 iterations. You can try focus on the z axis first (by adjusting the reward function on this)

### Visualizations
#### Agent (Quadcopter)
In order to better debug how your agent is performing after training, you should visualize it. The plot can be as simple as a x, y, z versus time. Here’s an example:
```
import matplotlib.pyplot as plt
%matplotlib inline
							
plt.plot(results['time'], results['x'], label='x')
plt.plot(results['time'], results['y'], label='y')
plt.plot(results['time'], results['z'], label='z')
plt.legend()

```

You can also plot the z-axis (altitude) vs x-axis or even in 3D. Here's an example of static 3D plot:

    from mpl_toolkits.mplot3d import Axes3D
    %matplotlib notebook # Change to %matplotlib inline if dont work
    
    fig = plt.figure(figsize = (14,8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.scatter(results['x'], results['y'], results['z'])
  
![](https://i.imgur.com/2Jeeq3d.gif)


In addition, an animated plot can be performed using the following class:

```
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# Make sure to change from notebook to inline after your tests
%matplotlib notebook
import time

class AnimatedPlot():
    def __init__(self):
        """Initialize parameters"""
        self.X, self.Y, self.Z = [], [], []

        self.fig = plt.figure(figsize = (14,8))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot(self, task, i_episode=None):
        pose = task.sim.pose[:3]
        self.X.append(pose[0])
        self.Y.append(pose[1])
        self.Z.append(pose[2])
        self.ax.clear()
        if i_episode:
            plt.title("Episode {}".format(i_episode))

        if len(self.X)>1:
            self.ax.scatter(self.X[:-1], self.Y[:-1], self.Z[:-1], c='k', alpha=0.3)
        if task.sim.done and task.sim.runtime > task.sim.time:
            # Colision
            self.ax.scatter(pose[0], pose[1], pose[2], c='r', marker='*', linewidths=5)
        else:
            self.ax.scatter(pose[0], pose[1], pose[2], c='k', marker='s', linewidths=5)

        self.fig.canvas.draw()
        plt.pause(0.5)
    
```

In order to make it work, you have to create an instance of the class before the while loop and inside the while loop, you have to call the method plot. Here’s an example:

```
myplot = AnimatedPlot()
state = agent.reset_episode() # start a new episode
while True:
    action = agent.act(state) 
    next_state, reward, done = task.step(action)
    agent.step(action, reward, next_state, done)
    state = next_state
    

    # CALL THE METHOD plot(task)
    myplot.plot(task)

    if done:
        break
    
```

![](https://i.imgur.com/V5IATLa.gif)



#### Reward function
It is also best practices to visualize the reward function in terms of its inputs. Here's one example based on the default reward function provided in the source code:
```
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline


def map_function(reward_function, x, y, target_pos):
    R = pd.DataFrame(np.zeros([len(x), len(y)]), index=y, columns=x)
    for xx in x:
        for yy in y:
            R[xx][yy] = reward_function([xx, yy], target_pos)

    return R


reward_function = lambda pose, target_pos: 1.-.3*(abs(pose - target_pos)).sum()

x_range = np.round(np.arange(-10.0,10,0.1), 2)
z_range = np.round(np.arange(20,0,-0.1), 2)

target_pos = np.array([0, 10])

R = map_function(reward_function, x_range, z_range, target_pos)

ax = sns.heatmap(R)
ax.set_xlabel("Position X-axis")
ax.set_ylabel("Position Z-axis")
plt.show()
```

![](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/38140/1543196738/download.png)



This way it is easier to play with alternative possibilities. For example, instead of a linear distance we can use an Euclidean distance:

    x_range = np.round(np.arange(0.0,10,0.1), 2)
    z_range = np.round(np.arange(10,0,-0.1), 2)
    target_pos = [0, 10]
    
    eucl_distance = lambda a, b: np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) # Alternative: np.linalg.norm(a-b)
    reward_function = lambda pose, target_pos: np.clip(2.-.25*(eucl_distance(pose, target_pos)), -1, 1)
    
    R = map_function(reward_function, x_range, z_range, target_pos)
    
    ax = sns.heatmap(R)
    ax.set_xlabel("Position X-axis")
    ax.set_ylabel("Position Z-axis")
    plt.show()
    
![](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/38140/1543196899/download__25_.png)


