import copy
from time import sleep
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box,Discrete,Dict,Space
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import gif
import time
import os

class SnowDriftEnv(MultiAgentEnv):
    def __init__(self,config):
        self.player_num=config["player_num"]
        self.drift_num=config['drift_num']
        self.height=config['world_height']
        self.width=config['world_width']
        self.reward=config['reward']
        self.cost=config['cost']
        self.final_time=config['final_time']
        self.render_env=config['render']
        self.max_block_comp=config['max_block_comp']
        self.prosocial=config['prosocial']
        self.state=None
        self.observation_space=Dict(dict(observation=Box(low=0,high=1,shape=(self.player_num+2,self.height,self.width),dtype=np.int8),
                                action_mask=Box(low=0,high=1,shape=(6,),dtype=np.int8), #{UP,DOWN,LEFT,RIGHT,STAY,STAG,HARE}
                                ))
        self.action_space=Discrete(6)
        self.players=[]
        for i in range(self.player_num):
            self.players.append(f'player_{i+1}')
        self.render_count=0
        self.render_frame=[]
        self.dir_name=int(time.time())

    def reset(self):
        if self.render_env:
            if not os.path.exists(f'./gif/exp{self.dir_name}'):
                os.makedirs(f'./gif/exp{self.dir_name}')
            if len(self.render_frame)!=0:
                self.render_count+=1
                gif.save(self.render_frame,path=f'./gif/exp{self.dir_name}/gif{self.render_count}.gif',duration=500,unit='milliseconds',between='frames',loop=True)
                self.render_frame=[]
            self.render_state()
            self.render()
            return self.__obs__([i for i in range(self.player_num)])

        self.state=np.zeros((self.player_num+2,self.height,self.width),dtype=np.int8)
        self.player_pos=np.zeros((self.player_num,2),dtype=np.int8)
        #add block
        while True:
            for _ in range(self.max_block_comp):
                self.block_build(random.choice(['3S','3L','4S','4L']))
            if self.check_connectivity():
                break
            else:
                self.state=np.zeros((self.player_num+2, self.height,self.width),dtype=np.int8)
        #find empty grid
        empty_grid=np.where(self.state[-2,:,:]==0)
        empty_grid_num=len(empty_grid[0])

        #add position of players
        for i in range(self.player_num):
            grid_index=np.random.randint(0,empty_grid_num)
            self.player_pos[i]=[empty_grid[0][grid_index],empty_grid[1][grid_index]]
            self.state[i,empty_grid[0][grid_index],empty_grid[1][grid_index]]=1

        #add position of preys
        drift_pos=np.random.choice(empty_grid_num,size=(self.drift_num,),replace=False)
        for i in range(self.drift_num):
            self.state[-1,empty_grid[0][drift_pos[i]],empty_grid[1][drift_pos[i]]]=1
        
        self.time=0
        
        return self.__obs__([i for i in range(self.player_num)])

    def step(self,action_dict):
        assert self.state is not None, "must call reset() first!"
        
        self.time+=1

        avail_players=action_dict.copy()
        avail_players_id=list(range(self.player_num))

        for id in avail_players_id:
            potential_action=action_dict[self.players[id]]
            available_action=self.__actionmask__(id)
            if available_action[potential_action]==0: #if there is an invalid action, print warnings and turn this action to STAY
                print(f"Invalid action from player_{id+1}")
                print(self.state)
                print(self.terminal)
                print(potential_action)
                self.render()
                avail_players[self.players[id]]=4
       
        rewards={player:0 for player in avail_players}
        dones={player:False for player in avail_players}
        remove_drift_players=[]

        for id in avail_players_id:
            x=self.player_pos[id,0]
            y=self.player_pos[id,1]
            if avail_players[self.players[id]]==0:
                self.state[id,x,y]=0
                self.state[id,x-1,y]=1
                self.player_pos[id,0]-=1
            elif avail_players[self.players[id]]==1:
                self.state[id,x,y]=0
                self.state[id,x+1,y]=1
                self.player_pos[id,0]+=1
            elif avail_players[self.players[id]]==2:
                self.state[id,x,y]=0
                self.state[id,x,y-1]=1
                self.player_pos[id,1]-=1
            elif avail_players[self.players[id]]==3:
                self.state[id,x,y]=0
                self.state[id,x,y+1]=1
                self.player_pos[id,1]+=1
            elif avail_players[self.players[id]]==5:
                remove_drift_players.append(id)

        while len(remove_drift_players)!=0:#find who hunt the same hares
            same_drift_with_zero=[remove_drift_players[0]]#storing ids of agents
            zero_drift_pos=tuple(self.player_pos[remove_drift_players[0]])
            assert self.state[-1,zero_drift_pos[0],zero_drift_pos[1]]==1, "no hare here, invalid action"
            for i in range(len(remove_drift_players)-1,0,-1):
                if tuple(self.player_pos[remove_drift_players[i]])==zero_drift_pos:
                    same_drift_with_zero.append(remove_drift_players[i])
                    remove_drift_players.pop(i)
            remove_drift_players.pop(0)
            for player in rewards:
                rewards[player]+=self.reward
            for id in same_drift_with_zero:
                rewards[self.players[id]]-=(self.cost*1.0/len(same_drift_with_zero))
            self.state[-1,zero_drift_pos[0],zero_drift_pos[1]]=0

        if len(np.nonzero(self.state[-1,:,:])[0])==0 or self.time>=self.final_time:
            for player in dones:
                dones[player]=True
            dones['__all__']=True
        else:
            dones['__all__']=False

        if self.prosocial:
            avg_reward=sum(rewards.values())/self.player_num
            prosocial_agent = ['player_1', 'player_2', 'player_3']
            for i in prosocial_agent:
                rewards[i]=avg_reward
        
        return self.__obs__(avail_players_id), rewards, dones, {self.players[id]:{} for id in avail_players_id}

    def __obs__(self,playerids):
        return {self.players[id]:{
            #'observation':self.__obs_state__(id),
            'observation':self.state,
            'action_mask':self.__actionmask__(id),
        }for id in playerids}

    def __obs_state__(self,id):
        mystate=self.state.copy()
        mystate[:self.player_num-id,:,:]=self.state[id:self.player_num,:,:]
        mystate[self.player_num-id:self.player_num,:,:]=self.state[:id,:,:]
        return mystate

    def __actionmask__(self,id):
        actions=np.zeros(6)
        x,y=self.player_pos[id,0],self.player_pos[id,1] #find current position
        if x==0 or self.state[-2,x-1,y]==1:
            actions[0]=1
        if x==self.height-1 or self.state[-2,x+1,y]==1:
            actions[1]=1
        if y==0 or self.state[-2,x,y-1]==1:
            actions[2]=1
        if y==self.width-1 or self.state[-2,x,y+1]==1:
            actions[3]=1
        if self.state[-1,x,y]==0:#not allowed to hunt hare
            actions[5]=1
        return 1-actions   #available actions---output 1
    
    def set_state(self,state_dict):
        state_dict_copy=copy.deepcopy(state_dict)
        self.state=state_dict_copy['state']
        self.time=state_dict_copy['time']
        self.player_pos=state_dict_copy['player_pos']
        return None

    def get_state(self):
        state_dict={}
        state_dict['state']=self.state.copy()
        state_dict['time']=self.time
        state_dict['player_pos']=self.player_pos.copy()
        return state_dict

    def block_build(self,mode):
        if mode=='3L':
            y=random.randint(0,self.width-2)
            x=random.randint(0,self.height-2)
            block_list=[(x,y),(x,y+1),(x+1,y),(x+1,y+1)]
            block_list.pop(random.randint(0,3))

        elif mode=='3S':
            direction=random.randint(0,1)
            if direction==0:
                y=random.randint(0,self.width-3)
                x=random.randint(1,self.height-2)
                block_list=[(x,y),(x,y+1),(x,y+2)]
            else:
                y=random.randint(1,self.width-2)
                x=random.randint(0,self.height-3)
                block_list=[(x,y),(x+1,y),(x+2,y)]

        elif mode=='4L':
            direction=random.randint(0,1)
            if direction==0:
                y=random.randint(0,self.width-3)
                x=random.randint(0,self.height-2)
                block_list=[(x,y),(x,y+1),(x,y+2),(x+1,y),(x+1,y+1),(x+1,y+2)]
            else:
                y=random.randint(0,self.width-2)
                x=random.randint(0,self.height-3)
                block_list=[(x,y),(x+1,y),(x+2,y),(x,y+1),(x+1,y+1),(x+2,y+1)]
            pop_list=[(0,1),(1,2),(3,4),(4,5)]
            pop_term=random.choice(pop_list)
            block_list.pop(pop_term[0])
            block_list.pop(pop_term[1]-1)

        elif mode=='4S':
            direction=random.randint(0,1)
            if direction==0:
                y=random.randint(0,self.width-4)
                x=random.randint(1,self.height-2)
                block_list=[(x,y),(x,y+1),(x,y+2),(x,y+3)]
            else:
                y=random.randint(1,self.width-2)
                x=random.randint(0,self.height-4)
                block_list=[(x,y),(x+1,y),(x+2,y),(x+3,y)]
        
        for block in block_list:
            if self.state[-2,block[0],block[1]]==1:
                return False
        for block in block_list:
            self.state[-2,block[0],block[1]]=1
        return True

    def check_connectivity(self):
        check_list=self.state[-2,:,:].copy()
        empty_grid=np.where(self.state[-2,:,:]==0)
        empty_grid_num=len(empty_grid[0])
        start_index=np.random.randint(0,empty_grid_num)
        start=(empty_grid[0][start_index],empty_grid[1][start_index])
        grid_queue=set()
        grid_queue.add(start)
        while len(grid_queue)!=0:
            pop_item=grid_queue.pop()
            x=pop_item[0]
            y=pop_item[1]
            if x!=0 and check_list[x-1,y]!=1:
                grid_queue.add((x-1,y))
            if y!=0 and check_list[x,y-1]!=1:
                grid_queue.add((x,y-1))
            if x!=self.height-1 and check_list[x+1,y]!=1:
                grid_queue.add((x+1,y))
            if y!=self.width-1 and check_list[x,y+1]!=1:
                grid_queue.add((x,y+1))
            check_list[x,y]=1
        if len(np.where(check_list==0)[0])!=0:
            return False
        return True

    def render(self):
        @gif.frame
        def plot():
            vlines=np.linspace(-0.5,-0.5+self.width,self.width+1)
            hlines=np.linspace(-0.5,-0.5+self.height,self.height+1)
            plt.hlines(hlines,-0.5,-0.5+self.width)
            plt.vlines(vlines,-0.5,-0.5+self.height)
            plt.axis('off')
            
            block_pos=np.nonzero(self.state[:,:,-3])
            stag_pos=np.nonzero(self.state[:,:,-2])
            hare_pos=np.nonzero(self.state[:,:,-1])
        
            for i in range(len(block_pos[0])):
                y=block_pos[0][i]
                x=block_pos[1][i]
                plt.fill([x+0.5,x+0.5,x-0.5,x-0.5],[y-0.5,y+0.5,y+0.5,y-0.5],'black')
            plt.scatter(stag_pos[1],stag_pos[0],s=200,c='green',marker='^')
            plt.scatter(hare_pos[1],hare_pos[0],s=200,c='red',marker='o')

            for i in range(self.player_num):
                if self.terminal[i]==0:
                    plt.text(self.player_pos[i][1]+0.2*math.cos(math.pi*2*i/self.player_num),self.player_pos[i][0]+0.2*math.sin(math.pi*2*i/self.player_num),f'P{i+1}')
        self.render_frame.append(plot())
        return True

    def render_show(self):
        vlines=np.linspace(-0.5,-0.5+self.width,self.width+1)
        hlines=np.linspace(-0.5,-0.5+self.height,self.height+1)
        plt.hlines(hlines,-0.5,-0.5+self.width)
        plt.vlines(vlines,-0.5,-0.5+self.height)
        plt.axis('off')
        plt.axis('square')
        block_pos=np.nonzero(self.state[:,:,-3])
        stag_pos=np.nonzero(self.state[:,:,-2])
        hare_pos=np.nonzero(self.state[:,:,-1])
    
        for i in range(len(block_pos[0])):
            y=block_pos[0][i]
            x=block_pos[1][i]
            plt.fill([x+0.5,x+0.5,x-0.5,x-0.5],[y-0.5,y+0.5,y+0.5,y-0.5],'black')
        plt.scatter(stag_pos[1],stag_pos[0],s=200,c='green',marker='^')
        plt.scatter(hare_pos[1],hare_pos[0],s=200,c='red',marker='o')

        for i in range(self.player_num):
            if self.terminal[i]==0:
                plt.text(self.player_pos[i][1]+0.1*math.cos(math.pi*2*i/self.player_num),self.player_pos[i][0]+0.1*math.sin(math.pi*2*i/self.player_num),f'P{i+1}')
        plt.show()
        return True
    
    def render_state(self):
        state_dict={}
        # 4p2s
        '''
        player_pos=[(3,4),(0,5),(3,6),(7,2)]
        stag_pos=[(2,5),(6,0)]
        hare_pos=[(1,6),(5,5),(6,1),(7,2)]
        block_pos=[(0,4),(1,4),(1,3),(1,2),(4,5),(4,6),(4,7),(5,7),(4,3),(5,3),(6,3),(7,3)]
        '''
        # 4p2s 19x19
        
        player_pos=[(3,14),(0,15),(3,16),(17,2)]
        stag_pos=[(2,15),(16,0)]
        hare_pos=[(1,16),(5,15),(16,1),(17,2)]
        block_pos=[(0,14),(1,14),(1,13),(1,12),(4,15),(4,16),(4,17),(5,17),(14,3),(15,3),(16,3),(17,3),(7,7),(6,7),(5,7),(5,6)]
        
        # 3p2s-1.1
        '''
        player_pos=[(1,3),(2,2),(7,5)]
        stag_pos=[(1,4),(4,1)]
        hare_pos=[(0,1),(0,3),(7,7)]
        block_pos=[(0,6),(1,6),(2,6),(3,6),(3,5),(3,4),(3,3),(3,2),(5,3),(5,4),(5,5),(6,3),(6,4),(7,4)]
        '''
        
        # 3p2s-1.2
        '''
        player_pos=[(4,2),(3,0),(1,6)]
        stag_pos=[(6,1),(0,2)]
        hare_pos=[(7,3),(7,7),(7,0)]
        block_pos=[(0,5),(1,5),(2,5),(3,5),(1,1),(1,2),(1,3),(2,3),(3,3),(4,3),(5,3),(5,4),(6,4),(7,4)]
        '''
        # 3p2s-2
        '''
        player_pos=[(2,4),(4,4),(0,4)]
        stag_pos=[(0,2),(4,6)]
        hare_pos=[(0,0),(7,3),(6,5)]
        block_pos=[(0,5),(1,5),(2,5),(3,5),(1,1),(1,2),(1,3),(2,3),(3,3),(4,3),(5,3),(5,4),(6,4),(7,4)]
        '''

        state=np.zeros((self.height,self.width,len(player_pos)+3),dtype=np.int8)
        for i in range(len(player_pos)):
            state[player_pos[i][0],player_pos[i][1],i]=1
        for pos in block_pos:
            state[pos[0],pos[1],-3]=1
        for pos in stag_pos:
            state[pos[0],pos[1],-2]=1
        for pos in hare_pos:
            state[pos[0],pos[1],-1]=1

        state_dict['state']=state
        state_dict['time']=0
        state_dict['first_hunt_time']=0
        state_dict['player_pos']=np.array(player_pos)
        state_dict['terminal']=np.array([0 for _ in range(len(player_pos))])

        self.set_state(state_dict)
        return None
