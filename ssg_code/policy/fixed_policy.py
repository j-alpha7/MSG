from ray.rllib.policy.policy import Policy
import numpy as np
import random

class RandomPolicy(Policy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def compute_actions(self,obs_batch,*args,**kwargs):
        action_batch=[]
        for obs in obs_batch:
            action_batch.append(
                np.random.choice(np.flatnonzero(obs[:6]))
            )
        return action_batch,[],{}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

class FixPolicy(Policy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def compute_actions(self,obs_batch,*args,**kwargs):
        action_batch=[]
        for obs in obs_batch:
            action_batch.append(
                np.random.choice(np.flatnonzero(obs[:5]))
            )
        return action_batch,[],{}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

class NearestDriftPolicy(Policy):
    def __init__(self, observation_space,action_space,config,*args, **kwargs):
        super().__init__(observation_space,action_space,config,*args, **kwargs)
        self.input_channels=config['input_channels']
        self.world_height=config['world_height']
        self.world_width=config['world_width']
        self.id=config['id']

    def compute_actions(self,obs_batch,*args,**kwargs):
        act_batch=[]
        for obs_flatten in obs_batch:
            obs=obs_flatten[6:].reshape((self.input_channels,self.world_height,self.world_width))
            cur_pos=np.where(obs[self.id-1,:,:]==1)
            x,y=cur_pos[0][0],cur_pos[1][0]
            drift_pos=np.where(obs[-1,:,:]==1)

            if len(drift_pos[0])==0:
                act_batch.append(np.random.choice(6))
            elif obs[-1,x,y]==1:
                act_batch.append(5)
            else:
                mindist=self.world_height*self.world_width
                argmindist_stack=[]
                for i in range(len(drift_pos[0])):
                    dist=abs(drift_pos[0][i]-x) + abs(drift_pos[1][i]-y)
                    if dist <mindist:
                        argmindist_stack=[(drift_pos[0][i],drift_pos[1][i])]
                        mindist=dist
                    elif dist==mindist:
                        argmindist_stack.append((drift_pos[0][i],drift_pos[1][i]))
                    
                argmindist=random.choice(argmindist_stack)
                x_goal,y_goal=argmindist[0],argmindist[1]
                action_list=[]
                if x_goal<x:
                    action_list.append(0)
                if y_goal<y:
                    action_list.append(2)
                if x_goal>x:
                    action_list.append(1)
                if y_goal>y:
                    action_list.append(3)
                act_batch.append(random.choice(action_list))

        return act_batch,[],{}
        
    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
