import sys
sys.path.append('..')

import os
import numpy as np
import copy
from planning.mcts_moa_relative import Node, RootParentNode,MCTS
from utils.ranked_reward import RankedRewardsBuffer
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from model.moa_model_relative import MOAModel

import multiprocessing as mp
from multiprocessing import Pool

import time as tm
import gif
import matplotlib.pyplot as plt
import math
torch, _ = try_import_torch()


class AlphaZeroPolicy(TorchPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        config,
        model,
        loss,
        action_distribution_class,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            config,
            model=model,
            loss=loss,
            action_distribution_class=action_distribution_class,
        )
        # we maintain an env copy in the policy that is used during mcts
        # simulations
        self.obs_space = observation_space

        if config['ToM_config']['my_id']<=config['env_config']['player_num']:
            self.player_num=config['env_config']['player_num']
            self.world_height=config['env_config']['world_height']
            self.world_width=config['env_config']['world_width']
            self.drift_num=config['env_config']['drift_num']
            self.reward=config['env_config']['reward']
            self.env_config=config['env_config'].copy()
            self.env_config['render'] = False
            self.tree_num=config['ToM_config']['tree_num']

            self.shovel_buffer_len=config['moa_config']['shovel_buffer_capacity']
            self.Q_temp=config['ToM_config']['Q_temperature']
            self.my_id=config['ToM_config']['my_id']
            self.gamma=config['ToM_config']['gamma']

            self.mcts=MCTS(model,config['mcts_config'],config['ToM_config']['evaluation'])

            self.model_list=[MOAModel(config['model']['custom_model_config']) for _ in range(self.player_num-1)]
            self.model_id_list=list(range(self.player_num))
            self.model_id_list.pop(self.my_id-1)
            player_name=[f'player_{i+1}' for i in self.model_id_list]
            self.model_dict=dict(zip(player_name,self.model_list))
            self.network_optimizer_list = [optim.Adam(m.parameters(), 5e-4) for m in self.model_list]

            if config['moa_config']['load_dir'] is not None:
                for i in range(self.player_num-1):
                    self.model_list[i].load_state_dict(torch.load(config['moa_config']['load_dir']+f'/player_{self.my_id}_to_{self.model_id_list[i]+1}.pth'))


            self.model_buffer=[[[],[],[],[]] for _ in range(self.player_num-1)]
            self.model_buffer_count=[0]*(self.player_num-1)
            #self.old_model_buffer=[[] for _ in range(self.player_num-1)]

            self.moa_batch_size=config['moa_config']['moa_batch_size']
            self.moa_update_time=config['moa_config']['moa_update_time']
            self.moa_buffer_capacity=config['moa_config']['moa_buffer_capacity']

            self.save_dir=config['moa_config']['save_dir']
            #os.makedirs(self.save_dir)

            self.env_creator=config['ToM_config']['env_creator']
            self.env=self.env_creator(self.env_config)
            self.env.reset()
            self.env_list=[self.env_creator(self.env_config) for _ in range(self.tree_num)]
            self.count=0
            #initialize the prior

            self.discount_factor = config['ToM_config']['discount_factor']

            self.times_prob = self.make_initial_times_prob()
            '''
            self.times_prob = [[3.0500823e-01, 3.7983122e-01, 2.3245916e-01, 6.8598598e-02, 1.3455566e-02, 1.4051415e-04, 5.0706393e-04],
[1.1879620e-01, 2.5998613e-01, 2.7968147e-01, 2.1755984e-01, 1.1031785e-01, 9.7626923e-03, 3.8958802e-03],
[3.7469736e-01, 3.7816498e-01, 1.9013903e-01, 4.7849841e-02, 8.3678309e-03, 7.5733481e-04, 2.3773868e-05],
[2.5270706e-01, 3.9299512e-01, 2.5010234e-01, 9.0962537e-02, 1.3066972e-02, 1.4264313e-04, 2.3773868e-05]]
            '''

            self.times_prob = [[1.11190714e-01, 3.41755390e-01, 3.62501711e-01, 1.46586984e-01, 3.40981185e-02, 3.85756046e-03, 9.60857869e-06],
[1.5158664e-01, 3.4380352e-01, 3.1232917e-01, 1.5747784e-01, 3.2855351e-02, 1.9461971e-03, 1.2505922e-06],
[1.1485257e-01, 3.2952741e-01, 3.2381171e-01, 1.8977621e-01, 4.1648317e-02, 3.8399841e-04, 5.8634491e-08],
[3.2692793e-01, 4.2309096e-01, 2.0245555e-01, 3.9553203e-02, 7.9718512e-03, 3.5180639e-07, 5.8634491e-08],]

            self.times_prob.pop(self.my_id - 1)
            self.times_prob = np.array(self.times_prob, dtype=np.float32)


            self.dir_name=int(tm.time())
            self.render_count=0

    @override(TorchPolicy)
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        input_dict = {"obs": obs_batch}
        if prev_action_batch is not None:
            input_dict["prev_actions"] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict["prev_rewards"] = prev_reward_batch

        return self.compute_actions_from_input_dict(
            input_dict=input_dict,
            episodes=episodes,
            state_batches=state_batches,
        )


    @override(Policy)
    def compute_actions_from_input_dict(
        self, input_dict, explore=None, timestep=None, episodes=None, state_batches=None, **kwargs
    ):
        #self.Q_temp=min(1+(7-1)*1e-6*timestep,7)
        with torch.no_grad():
            action_batch=[]
            new_state_batch=[]
            obs_batch=input_dict['obs']
            for n in range(len(obs_batch)):
                obs_flatten=obs_batch[n]
                time=episodes[n].length
                obs=obs_flatten[6:].reshape((self.player_num+2,self.world_height,self.world_width))
                now_state_dict=self.set_env_states(obs,time)

                t1=tm.time()

                if time==0: #initialize new_state at the first step of an episode
                    now_drift_pos=self.get_drift_pos(obs)
                    self.prob=np.full((self.player_num-1, self.drift_num),1/self.drift_num, dtype=np.float32)
                    episodes[n].user_data[f"last_player_pos{self.my_id}"]=[]
                    self.shovel_times = [0]*(self.player_num - 1)
                    #self.times_prob = self.make_initial_times_prob()
                    self.shovel_prob = 1 - self.times_prob[:,0]

                else: # new_state when t>1
                    now_drift_pos=episodes[n].user_data[f"last_drift_pos{self.my_id}"]
                    exist_drift=[]
                    for i,pos in enumerate(now_drift_pos):
                        if pos[0]!=-1:
                            if obs[-1,pos[0],pos[1]]==0:
                                pos[0]=pos[1]=-1
                            else:
                                exist_drift.append(i)
                    last_action=episodes[n]._agent_to_last_action
                    last_obs=episodes[n].user_data[f"last_obs{self.my_id}"]
                    last_player_pos=episodes[n].user_data[f"last_player_pos{self.my_id}"][-1]
                    for i,player_id in enumerate(self.model_id_list):
                        player_name=f'player_{player_id+1}'
                        if last_action[player_name] == 5:
                            self.shovel_times[i] += 1
                            self.shovel_prob[i] -= self.times_prob[i, self.shovel_times[i]]
                        avail_drift_layer=[None]*self.drift_num
                        for drift_id in exist_drift:
                            avail_drift_layer[drift_id]=now_drift_pos[drift_id]-last_player_pos[player_id]
                        prob_likelihood=self.model_dict[player_name].get_action_prob(last_obs[player_name], time-1, last_action[player_name],avail_drift_layer)
                        self.prob[i]*=prob_likelihood
                        if (self.prob[i]==0).all():
                            for drift_id in exist_drift:
                                if self.prob[i, drift_id]==0:
                                    self.prob[i, drift_id] = 1
                        self.prob[i]/=(np.sum(self.prob[i]))
                #choose actions according to belief
                t2=tm.time()

                self.env.set_state(now_state_dict)
                episodes[n].user_data[f"last_obs{self.my_id}"]=self.env.__obs__(list(range(self.player_num)))
                episodes[n].user_data[f"last_player_pos{self.my_id}"].append(now_state_dict['player_pos'])
                tree_node_list=[]
                prob_list=[]
                original_obs={'observation':obs, 'action_mask':obs_flatten[:6]}

                target_prefernce_list=self.sample_target_preference(self.shovel_prob,self.tree_num)

                for tree_index in range(self.tree_num):
                    tree_node_list.append(
                        Node(
                        state=now_state_dict,
                        obs=original_obs,
                        reward=0,
                        done=False,
                        action=None,
                        #parent=RootParentNode(env=self.env),
                        parent=RootParentNode(env=self.env_list[tree_index]),
                        mcts=self.mcts,
                        model_list=self.model_dict,
                        model_id_list=self.model_id_list,
                        target_prey=target_prefernce_list[tree_index],
                        drift_num=self.drift_num,
                        drift_pos=now_drift_pos,
                        id=self.my_id,
                        count=0,
                        gamma=self.gamma
                        )
                        )

                if self.tree_num==1:
                    mcts_policy, action, tree_node, Q_value = self.mcts.compute_action(tree_node_list[0])
                else:
                    p=Pool(self.tree_num)
                    action_info_list=p.map(self.mcts.compute_action, tree_node_list)
                    p.close()
                    p.join()
                    # mcts_policy, action, tree_node,Q_value = self.mcts.compute_action(tree_node)
                    Q_value_total=np.zeros((6,))
                    '''
                    for list_index, tree_index in enumerate(actual_calculate_tree):
                        Q_value_total+=(action_info_list[list_index][3]*prob_list[tree_index])
                    '''
                    for tree_id in range(self.tree_num):
                        Q_value_total+=(action_info_list[tree_id][3])
                    Q_value_total/=self.tree_num
                    Q_value_total-=(np.max(Q_value_total))

                    #mcts_policy=np.power(Q_value_total/np.max(Q_value_total),self.mcts.temperature)
                    mcts_policy=np.exp(Q_value_total*self.Q_temp)
                    mcts_policy=mcts_policy/np.sum(mcts_policy)
                    action=np.argmax(mcts_policy)
                    #if self.my_id == 1: print(Q_value_total)
                    #action=np.random.choice(6,p=mcts_policy)

                t3=tm.time()
                action_batch.append(action)
                episodes[n].user_data[f"last_drift_pos{self.my_id}"]=now_drift_pos
                if time == 0:
                    episodes[n].user_data[f"mcts_policies{self.my_id}"] = [mcts_policy]
                else:
                    episodes[n].user_data[f"mcts_policies{self.my_id}"].append(mcts_policy)

        self.count+=1
        #print discounted times for agents to hunt stags/hares

        if self.my_id<3 and self.count%201==0:
            print(self.times_prob)
            print(self.my_id)
            #print(t2-t1)
            #print(t3-t2)
            print('BBB')
            #for tree_id in range(self.tree_num):
            #    print(action_info_list[tree_id][3])

        return action_batch,np.array(new_state_batch).T,{}

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        '''
        if self.my_id==1 and self.render_count % 5==0:
            self.save_gif(sample_batch['obs'])
        '''
        self.render_count+=1
        for i in range(self.player_num-1):
            model_id=self.model_id_list[i]
            state_set=other_agent_batches[f'player_{model_id+1}'][1]['obs']
            action_set=other_agent_batches[f'player_{model_id+1}'][1]['actions']
            player_pos_set=episode.user_data[f"last_player_pos{self.my_id}"]
            target_pos=-np.ones((2,),dtype=np.int8)
            length=0
            shovel_count = 0

            if action_set[-1]==5:
                target_pos=self.get_target_pos(state_set[-1,6:], model_id)
                target_list=[target_pos-player_pos_set[-1][model_id]]
                shovel_count+=1
                length=len(action_set)
            else:
                target_list=[None]
            for j in range(len(action_set)-2,-1,-1):
                if action_set[j]==5:
                    target_pos=self.get_target_pos(state_set[j,6:], model_id)
                    target_list.append(target_pos-player_pos_set[j][model_id])
                    shovel_count+=1
                    length=length or j+1
                else:
                    if target_pos[0]==-1:
                        target_list.append(None)
                    else:
                        target_list.append(target_pos-player_pos_set[j][model_id])
            target_list.reverse()
            self.times_prob[i] *= self.discount_factor
            self.times_prob[i,shovel_count] += (1-self.discount_factor)
            if length>0:
                self.model_buffer[i][0].append(state_set[:length, :].copy())
                self.model_buffer[i][1].append(action_set[ :length].copy())
                self.model_buffer[i][2].append(np.array(list(range(length))))
                self.model_buffer[i][3].append(np.array(target_list[:length]))
                assert length==len(action_set) or target_list[length]==None, target_list
                self.model_buffer_count[i]+=length

                if self.model_buffer_count[i]>=self.moa_buffer_capacity:
                    self.moa_update(i)

        # add mcts policies to sample batch
        sample_batch["mcts_policies"] = np.array(episode.user_data[f"mcts_policies{self.my_id}"])[
            sample_batch["t"]
        ]

        # sample_batch["value_label"] = final_reward * np.ones_like(sample_batch["t"])
        sample_batch["value_label"] = copy.deepcopy(sample_batch["rewards"])
        for i in range(len(action_set)-2,-1,-1):
            sample_batch["value_label"][i] += (sample_batch["value_label"][i+1] * self.gamma)

        return sample_batch

    @override(TorchPolicy)
    def learn_on_batch(self, postprocessed_batch):
        train_batch = self._lazy_tensor_dict(postprocessed_batch)

        loss_out, policy_loss, value_loss = self._loss(
            self, self.model, self.dist_class, train_batch
        )
        self._optimizers[0].zero_grad()
        loss_out.backward()

        grad_process_info = self.extra_grad_process(self._optimizers[0], loss_out)
        self._optimizers[0].step()

        grad_info = self.extra_grad_info(train_batch)
        grad_info.update(grad_process_info)
        grad_info.update(
            {
                "total_loss": loss_out.detach().cpu().numpy(),
                "policy_loss": policy_loss.detach().cpu().numpy(),
                "value_loss": value_loss.detach().cpu().numpy(),
            }
        )

        return {LEARNER_STATS_KEY: grad_info}

    def set_env_states(self,obs,time): #set the state of the environments
        state_dict={}
        env_state=obs.copy()

        player_pos=np.zeros((self.player_num,2),dtype=np.int8)
        cur_pos=np.nonzero(obs)
        for i in range(self.player_num):
            assert cur_pos[0][i]==i, 'error'
            player_pos[i]=[cur_pos[1][i],cur_pos[2][i]]

        state_dict['state']=env_state
        state_dict['time']=time
        state_dict['player_pos']=player_pos

        return state_dict

    def make_initial_times_prob(self):
        times = [0]
        factor = 1
        for i in range(1, self.drift_num + 1):
            factor *= i
            times.append(1/factor)
        times = np.array(times, dtype=np.float32)*np.exp(-1)
        times[0] = 1 - np.sum(times[1:])
        times_prob = np.array([times for _ in range(self.player_num - 1)], dtype=np.float32)
        return times_prob

    def get_drift_pos(self,obs):
        drift_pos=np.zeros((self.drift_num,2),dtype=np.int8)
        cur_pos=np.nonzero(obs[-1,:,:])
        for i in range(self.drift_num):
            drift_pos[i]=[cur_pos[0][i],cur_pos[1][i]]
        return drift_pos

    def sample_target_preference(self, shovel_prob, num):
        prob=np.zeros((self.player_num-1,self.drift_num+1),dtype=np.float32)
        for i in range(self.player_num-1):
            prob[i]=np.append(self.prob[i]*shovel_prob[i],1-shovel_prob[i])
        prob=prob/(np.sum(prob,axis=1)[...,None])
        prob+=1e-30
        if (prob<0).any():
            prob[prob<0] = 1e-31
            for i in range(self.player_num-1):
                assert(np.abs(np.sum(prob[i])-1)<1e-5)
            prob=prob/(np.sum(prob,axis=1)[...,None])
        target_preference_list=[np.zeros((self.player_num, self.drift_num+1),dtype=np.int8) for _ in range(num)]
        for i in range(self.player_num - 1):
            '''
            if (prob[i]<0).any():
                print(prob[i])
            '''
            for k in range(num):
                target_preference_list[k][i]=np.random.choice(self.drift_num+1,size=(self.drift_num+1,),replace=False,p=prob[i])
        return target_preference_list

    def get_target_pos(self, obs_flatten, player):
        obs=obs_flatten.reshape((self.player_num+2,self.world_height,self.world_width))
        pos=np.nonzero(obs[player,:,:])
        assert len(pos[0])==1,'error'

        return np.array([pos[0][0],pos[1][0]])

    def moa_update(self,i):
        print(self.my_id)
        print(self.model_id_list[i]+1)
        print('AAA')

        state=torch.from_numpy(np.concatenate(self.model_buffer[i][0],axis=0)).float()
        action=F.one_hot( torch.from_numpy( np.concatenate(self.model_buffer[i][1],axis=0, dtype = np.int64) ), 6 ).float()
        time=torch.from_numpy(np.concatenate(self.model_buffer[i][2],axis=0)).float()
        target=torch.from_numpy( np.concatenate(self.model_buffer[i][3],axis=0) ).float()

        if os.path.exists(self.save_dir+f'/player_{self.my_id}_to_{self.model_id_list[i]+1}.pth'):
            for _ in range(10):
                try:
                    self.model_list[i].load_state_dict(torch.load(self.save_dir+f'/player_{self.my_id}_to_{self.model_id_list[i]+1}.pth'))
                    break
                except EOFError:
                    time.sleep(2)

        for _ in range(self.moa_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(self.model_buffer_count[i])), self.moa_batch_size, False):
                action_dist=self.model_list[i].forward(state[index], time[index], target[index])
                loss=-torch.sum(action[index]*torch.log(action_dist+1e-30),dim=1).mean()
                self.network_optimizer_list[i].zero_grad()
                loss.backward()
                self.network_optimizer_list[i].step()

        torch.save(self.model_list[i].state_dict(),self.save_dir+f'/player_{self.my_id}_to_{self.model_id_list[i]+1}.pth')

        self.model_buffer[i]=[[],[],[],[]]
        self.model_buffer_count[i]=0

    def save_gif(self, obs_flatten):
        obs=obs_flatten[:,6:].reshape((obs_flatten.shape[0], self.player_num+2,self.world_height,self.world_width))
        render_frame=[]
        @gif.frame
        def plot(state):
            pos=np.nonzero(state[:self.player_num])
            player_pos=np.zeros((self.player_num, 2))
            for i in range(self.player_num):
                player_pos[i]=[pos[1][i], pos[2][i]]
            vlines=np.linspace(-0.5,-0.5+self.world_width,self.world_width+1)
            hlines=np.linspace(-0.5,-0.5+self.world_height,self.world_height+1)
            plt.hlines(hlines,-0.5,-0.5+self.world_width)
            plt.vlines(vlines,-0.5,-0.5+self.world_height)
            plt.axis('off')

            block_pos=np.nonzero(state[-2,:,:])
            drift_pos=np.nonzero(state[-1,:,:])

            for i in range(len(block_pos[0])):
                y=block_pos[0][i]
                x=block_pos[1][i]
                plt.fill([x+0.5,x+0.5,x-0.5,x-0.5],[y-0.5,y+0.5,y+0.5,y-0.5],'black')
            plt.scatter(drift_pos[1],drift_pos[0],s=200,c='red',marker='o')

            for i in range(self.player_num):
                plt.text(player_pos[i][1]+0.2*math.cos(math.pi*2*i/self.player_num),player_pos[i][0]+0.2*math.sin(math.pi*2*i/self.player_num),f'P{i+1}')
        for i in range(obs_flatten.shape[0]):
            render_frame.append(plot(obs[i]))
        if not os.path.exists(f'./gif/exp{self.dir_name}'):
            os.makedirs(f'./gif/exp{self.dir_name}')
        gif.save(render_frame,path=f'./gif/exp{self.dir_name}/gif{self.render_count}.gif',duration=500,unit='milliseconds',between='frames',loop=True)
