from ray.rllib.policy.policy import Policy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple
from itertools import count
from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
import os
import time as tm
from ray.rllib.utils.annotations import override

Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'action_dist'])

def load(model,path):
    if os.path.exists(path):
        for _ in range(10):
            try:
                model.load_state_dict(torch.load(path))
                break
            except EOFError:
                tm.sleep(2)
            except RuntimeError:
                tm.sleep(2)

def random_choice(obs_batch):
    action_batch=[]
    for obs in obs_batch:
        action_batch.append(
            np.random.choice(np.flatnonzero(obs['action_mask'][:5]))
        )
    return  action_batch, None, None

class Mymodel(nn.Module):
    def __init__(self,model_config):
        super(Mymodel, self).__init__()
        self.lstm_state_size=model_config['lstm_state_size']
        self.input_channels=model_config['input_channels']
        self.world_height=model_config['world_height']
        self.world_width=model_config['world_width']
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.shared_layer=nn.Sequential(
            nn.Conv2d(self.input_channels,16,3,1,1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,1,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*self.world_height*self.world_width,512),
            nn.ReLU()
        ).to(self.device)
        self._action_branch=nn.Linear(512,6).to(self.device)
        self._value_branch=nn.Linear(512,1).to(self.device)
        self._value_out=None

    def forward(self, input_dict):
        x=input_dict['observation'].to(self.device).float()
        action_mask=input_dict['action_mask'].to(self.device).float()
        x = self.shared_layer(x)
        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        action_prob = F.softmax(self._action_branch(x)+inf_mask, dim=1)
        self._value_out = self._value_branch(x)
        return action_prob.cpu()

    def value_function(self):
        return self._value_out.cpu()

    def flatten_state_forward(self,state_flatten):
        x=state_flatten[:,6:].reshape(state_flatten.shape[0], self.input_channels, self.world_height, self.world_width)
        x=x.to(self.device).float()
        action_mask=state_flatten[:,:6].to(self.device).float()
        x = self.shared_layer(x)
        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        action_prob = F.softmax(self._action_branch(x)+inf_mask, dim=1)
        return action_prob.cpu()



class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 20
    buffer_capacity = 19999
    batch_size = 1024
    cur_kl_coef=0.2

    def __init__(self, config):
        super(PPO, self).__init__()
        self.network=Mymodel(config['model']['custom_model_config']).float()
        self.gamma=config['lola_config']['gamma']
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.network_optimizer = optim.Adam(self.network.parameters(), config['lola_config']['lr'])

    def select_action(self, states):
        input_states = {
            'observation':torch.from_numpy(
                np.stack( [state['observation'] for state in states] )
                ),
            'action_mask':torch.from_numpy(
                np.stack( [state['action_mask'] for state in states] )
                )
        }

        with torch.no_grad():
            action_prob = self.network.forward(input_states)
            c = Categorical(action_prob)
            action = c.sample()
        action_p = []
        for i in range(action.shape[0]):
            action_p.append(action_prob[i, action[i].item()].item())
        return action.numpy(), action_p, action_prob.numpy()

    def get_value(self, state):
        input_state={
            'observation':torch.from_numpy(state['observation']).unsqueeze(0),
            'action_mask':torch.from_numpy(state['action_mask'])
        }
        with torch.no_grad():
            self.network.forward(input_state)
            value = self.network.value_function()
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def store_transitions(self, transitions):
        self.buffer.extend(transitions)
        self.counter += len(transitions)

    def update(self, terminal_buffer):
        state = torch.tensor(np.array([t.state['observation'] for t in self.buffer]), dtype=torch.float)
        action_mask = torch.tensor(np.array([t.state['action_mask'] for t in self.buffer]), dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        old_action_dist = torch.tensor(np.array([t.action_dist for t in self.buffer]), dtype=torch.float).squeeze()

        R = 0
        Gt = []
        traj_len=0
        episode_count=1
        for r in reward[::-1]:
            traj_len+=1
            R = r + self.gamma * R
            Gt.insert(0, R)
            if traj_len==terminal_buffer[-episode_count]:
                traj_len=0
                episode_count+=1
                R=0
        assert episode_count==len(terminal_buffer)+1,'ERROR'
        
        Gt = torch.tensor(Gt, dtype=torch.float)
        #Gt = (Gt - Gt.mean())/(Gt.std()+1e-10)
        #print("The agent is updateing....")
        for _ in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                action_prob_dist = self.network.forward({'observation':state[index],'action_mask':action_mask[index]})
                action_prob = action_prob_dist.gather(1, action[index]) # new policy
                action_dist=Categorical(action_prob_dist)

                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.network.value_function()
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                
                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                #self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)

                value_loss = F.mse_loss(Gt_index, V)
                #self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)

                entropy_loss=action_dist.entropy().mean()
                #self.writer.add_scalar('loss/entropy_loss', entropy_loss, global_step=self.training_step)

                kl_loss=torch.kl_div(torch.log(action_prob_dist+1e-30),old_action_dist[index]).mean()
                #self.writer.add_scalar('loss/kl_loss', kl_loss, global_step=self.training_step)

                total_loss=action_loss+0.0001*value_loss#-0*entropy_loss+self.cur_kl_coef*kl_loss
                #self.writer.add_scalar('loss/total_loss', total_loss, global_step=self.training_step)

                self.network_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.network_optimizer.step()
                self.training_step += 1
                if kl_loss>0.005*2:
                    self.cur_kl_coef*=2
                elif kl_loss<0.005/2:
                    self.cur_kl_coef/=2

        del self.buffer[:] # clear experience

class LOLA():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 1
    buffer_capacity = 3999
    batch_size = 1024
    cur_kl_coef=0.2

    def __init__(self, config):
        super(LOLA, self).__init__()
        self.network=Mymodel(config['model']['custom_model_config']).float()
        self.id=config['lola_config']['id']
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.player_num=config['env_config']['player_num']
        self.player_name_list=[f'player_{i+1}' for i in range(self.player_num)]
        self.my_player_name=self.player_name_list[self.id-1]
        self.network_optimizer = optim.Adam(self.network.parameters(), config['lola_config']['lr'])
        self.gamma=config['lola_config']['gamma']

        self.env_num = config['lola_config']['env_num']
        env_creator=config['lola_config']['env_creator']
        self.env = [env_creator(config['env_config']) for _ in range(self.env_num)]
        for i in range(self.env_num):
            self.env[i].reset()

        self.opponent_model_name_list=self.player_name_list.copy()
        self.opponent_model_name_list.remove(self.my_player_name)
        self.opponent_model={name: Mymodel(config['model']['custom_model_config']).float() for name in self.opponent_model_name_list}
        self.opponent_optimizer = {name: optim.Adam(self.opponent_model[name].parameters(), config['lola_config']['lr']) for name in self.opponent_model}
        self.update_opponent_model={name: PPO(config) for name in self.opponent_model_name_list}
        for name, model in self.update_opponent_model.items():
            model.network.load_state_dict(self.opponent_model[name].state_dict())

        self.save_dir=config['lola_config']['save_dir']

    def select_action(self, states):
        input_states = {
            'observation':torch.from_numpy(
                np.stack( [state['observation'] for state in states] )
                ),
            'action_mask':torch.from_numpy(
                np.stack( [state['action_mask'] for state in states] )
                )
        }

        with torch.no_grad():
            action_prob = self.network.forward(input_states)
            c = Categorical(action_prob)
            action = c.sample()
        action_p = []
        for i in range(action.shape[0]):
            action_p.append(action_prob[i, action[i].item()].item())
        return action.numpy(), action_p, action_prob.numpy()

    def get_value(self, state):
        input_state={
            'observation':torch.from_numpy(state['observation']).unsqueeze(0),
            'action_mask':torch.from_numpy(state['action_mask'])
        }
        with torch.no_grad():
            self.network.forward(input_state)
            value = self.network.value_function()
        return value.item()

    def save_param(self):
        torch.save(self.network.state_dict(), f'./params/player_{self.id}.pth')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def store_transitions(self, transitions):
        self.buffer.extend(transitions)
        self.counter += len(transitions)

    def rollout(self, record_player_list):
        capacity_list=[]
        for player in record_player_list:
            if player == self.my_player_name:
                capacity_list.append(self.buffer_capacity)
            else:
                capacity_list.append(self.update_opponent_model[player].buffer_capacity)
        max_capacity = min(capacity_list)

        terminal_buffer_list=[{player_id:[] for player_id in record_player_list} for _ in range(self.env_num)]
        transition_list=[{player_id:[] for player_id in record_player_list} for _ in range(self.env_num)]
        episode_len_count=0
        buffer_full=False

        while not buffer_full:
            obs = [self.env[i].reset() for i in range(self.env_num)]
            dones = [{'__all__':False} for _ in range(self.env_num)]
            alive_env = list( range(self.env_num) )
            for t in count(): # 对一个episode的模拟
                action={}
                action_prob={}
                action_dist={}
                obses = {}
                
                for player in self.player_name_list:
                    obses[player] = [obs_of_one_env[player] for obs_of_one_env in obs]
                    if player==self.my_player_name:
                        action[player], action_prob[player], action_dist[player]=self.select_action(obses[player])
                    else:
                        action[player], action_prob[player], action_dist[player]= self.update_opponent_model[player].select_action(obses[player])
                        #action[player], action_prob[player], action_dist[player] = random_choice(obses[player])
                
                new_obses = []
                new_alive_env=[]
                rewards = {player: [] for player in self.player_name_list}
                for i,env_id in enumerate(alive_env):
                    new_obs, reward, done, _ = self.env[env_id].step(
                        {player: action[player][i] for player in self.player_name_list}
                        )
                    for player in reward:
                        rewards[player].append(reward[player])
                    dones[i]['__all__'] = done['__all__']
                    if not done['__all__']:
                        new_obses.append(new_obs)
                        new_alive_env.append(env_id)
                    else:
                        episode_len_count+=(t+1)
                        for player in record_player_list:
                            terminal_buffer_list[i][player].append(t+1) # 记录该episode长度

                for i, env_id in enumerate(alive_env):
                    for player in record_player_list:
                        transition_list[env_id][player].append(Transition(obses[player][i], 
                        action[player][i], action_prob[player][i], rewards[player][i], action_dist[player][i]))
            
                obs = new_obses
                alive_env = new_alive_env
                if len(alive_env)==0:
                    buffer_full= (episode_len_count > max_capacity)
                    break

        total_terminal_buffer_list = {player_id:[] for player_id in record_player_list}
        total_transition_list = {player_id:[] for player_id in record_player_list}
        for tbl_of_one_env in terminal_buffer_list:
            for player_id in record_player_list:
                total_terminal_buffer_list[player_id].extend(tbl_of_one_env[player_id])
        for tl_of_one_env in transition_list:
            for player_id in record_player_list:
                total_transition_list[player_id].extend(tl_of_one_env[player_id])
        
        return total_terminal_buffer_list, total_transition_list    

    def update(self):
        for name, model in self.update_opponent_model.items():
            model.network.load_state_dict(self.opponent_model[name].state_dict())
        load(self.network, self.save_dir+f'/{self.my_player_name}.pth')

        #imagine 1 update for other agents
        t1=tm.time()
        
        terminal_buffer_list, transition_list=self.rollout(self.opponent_model_name_list)
        for player, model in self.update_opponent_model.items():
            model.store_transitions(transition_list[player])
            model.update(terminal_buffer_list[player])
        
        t2=tm.time()
        #play with others' foresee policies
        #load(self.network, self.save_dir+f'/{self.my_player_name}.pth')
        terminal_buffer_list, transition_list=self.rollout([self.my_player_name])
        terminal_buffer = terminal_buffer_list[self.my_player_name]
        self.store_transitions(transition_list[self.my_player_name])
        
        t3=tm.time()
        state = torch.tensor(np.array([t.state['observation'] for t in self.buffer]), dtype=torch.float)
        action_mask = torch.tensor(np.array([t.state['action_mask'] for t in self.buffer]), dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        old_action_dist = torch.tensor(np.array([t.action_dist for t in self.buffer]), dtype=torch.float).squeeze()
        #print(len(self.buffer))

        R = 0
        Gt = []
        traj_len=0
        episode_count=1
        for r in reward[::-1]:
            traj_len+=1
            R = r + self.gamma * R
            Gt.insert(0, R)
            if traj_len==terminal_buffer[-episode_count]:
                traj_len=0
                episode_count+=1
                R=0
        assert episode_count==len(terminal_buffer)+1,'ERROR'
        '''
        for i,ac in enumerate(action):
            if ac == 5:
                print(Gt[i+1])
                print(Gt[i])
                print(Gt[i] - Gt[i+1] * self.gamma)
                print('-----------------------------------------')
        '''
        Gt = torch.tensor(Gt, dtype=torch.float)
        #Gt = (Gt - Gt.mean())/(Gt.std()+1e-10)
        #print("The agent is updateing....")

        #load(self.network, self.save_dir+f'/{self.my_player_name}.pth')
        for _ in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):

                action_prob_dist = self.network.forward({'observation':state[index],'action_mask':action_mask[index]})
                action_prob = action_prob_dist.gather(1, action[index]) # new policy
                action_dist=Categorical(action_prob_dist)

                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.network.value_function()
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                
                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                #self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)

                value_loss = F.mse_loss(Gt_index, V)
                #self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)

                entropy_loss = action_dist.entropy().mean()
                #self.writer.add_scalar('loss/entropy_loss', entropy_loss, global_step=self.training_step)

                kl_loss=torch.kl_div(torch.log(action_prob_dist+1e-30),old_action_dist[index]).mean()
                #self.writer.add_scalar('loss/kl_loss', kl_loss, global_step=self.training_step)

                total_loss=action_loss+0.0001*value_loss#-0*entropy_loss+self.cur_kl_coef*kl_loss
                #self.writer.add_scalar('loss/total_loss', total_loss, global_step=self.training_step)

                self.network_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.network_optimizer.step()
                self.training_step += 1
                if kl_loss>0.005*2:
                    self.cur_kl_coef*=2
                elif kl_loss<0.005/2:
                    self.cur_kl_coef/=2
        torch.save(self.network.state_dict(),self.save_dir+f'/{self.my_player_name}.pth')
        t4=tm.time()
        print(t2-t1)
        print(t3-t2)
        print(t4-t3)
        print('AAA')
        '''
        if kl_loss>0.01*2:
            self.cur_kl_coef*=2
        elif kl_loss<0.01/2:
            self.cur_kl_coef/=2
        '''

        del self.buffer[:] # clear experience

class LOLAPolicy(Policy):
    def __init__(self, observation_space,action_space,config,*args, **kwargs):
        super().__init__(observation_space,action_space,config,*args, **kwargs)
        if config['lola_config']['id']<=config['env_config']['player_num']:
            self.env_config=config['env_config']
            self.player_num=self.env_config['player_num']
            self.world_height=self.env_config['world_height']
            self.world_width=self.env_config['world_width']
            self.id=config['lola_config']['id']
            self.train=config['lola_config']['train']
            self.player=LOLA(config)

            self.memory_buffer={name: [[],[]] for name in self.player.opponent_model_name_list}
            self.memory_buffer_count={name:0 for name in self.player.opponent_model_name_list}
            self.buffer_capacity=config['lola_config']['buffer_capacity']
            self.moa_batch_size=config['lola_config']['moa_batch_size']
            self.moa_update_time=config['lola_config']['moa_update_time']

            self.save_dir=config['lola_config']['save_dir']
            self.load_dir=config['lola_config']['load_dir']
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
            if self.load_dir is not None:
                self.player.network.load_state_dict(torch.load(self.load_dir+f'/{self.player.my_player_name}.pth',self.device))
                for name, model in self.player.opponent_model.items():
                    model.load_state_dict(torch.load(self.load_dir+f'/{name}_in_{self.player.my_player_name}.pth',self.device))

    def compute_actions(self,obs_batch,*args,**kwargs):
        act_batch=[]
        with torch.no_grad():
            action_prob_batch=self.player.network.flatten_state_forward(torch.from_numpy(obs_batch))
            act_batch = Categorical(action_prob_batch).sample().numpy()
        return act_batch, [], {}

    def learn_on_batch(self, samples):
               
        for name,model in self.player.opponent_model.items():
            state=torch.from_numpy(samples[f'obs_{name}_in_{self.id}'].copy()).float()
            action=F.one_hot( torch.from_numpy(samples[f'actions_{name}_in_{self.id}'].copy()).to(torch.int64), 6).float()
            load(model, self.save_dir+f'/{name}_in_{self.player.my_player_name}.pth')
            for _ in range(self.moa_update_time):
                for index in BatchSampler(SubsetRandomSampler(range(samples[f'actions_{name}_in_{self.id}'].shape[0])), self.moa_batch_size, False):
                    action_dist=model.flatten_state_forward(state[index])
                    loss=-torch.sum(action[index]*torch.log(action_dist+1e-30),dim=1).mean()
                    self.player.opponent_optimizer[name].zero_grad()
                    loss.backward()
                    self.player.opponent_optimizer[name].step()
            torch.save(model.state_dict(),self.save_dir+f'/{name}_in_{self.player.my_player_name}.pth')
        
        #self.memory_buffer={name: [[],[]] for name in self.player.opponent_model_name_list}
        #self.memory_buffer_count={name:0 for name in self.player.opponent_model_name_list}
        print(self.id)
        self.player.update()

        return {LEARNER_STATS_KEY: {}}
          
        pass
        

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        '''
        if self.train:
            all_buffer_full=True
            for name in self.player.opponent_model_name_list:
                total_time=other_agent_batches[name][1]['actions'].shape[0]
                self.memory_buffer[name][0].append(other_agent_batches[name][1]['obs'].copy())
                self.memory_buffer[name][1].append(other_agent_batches[name][1]['actions'].copy())
                self.memory_buffer_count[name]+=total_time
            
                if self.memory_buffer_count[name]<self.buffer_capacity:
                    all_buffer_full=False
            if all_buffer_full:
                print(self.memory_buffer_count)
                self.update()
        '''
        
        if self.train:
            for name in self.player.opponent_model_name_list:
                sample_batch[f'obs_{name}_in_{self.id}']=other_agent_batches[name][1]['obs'].copy()
                sample_batch[f'actions_{name}_in_{self.id}']=other_agent_batches[name][1]['actions'].copy()
        
        return sample_batch

    def update(self):
        for name,model in self.player.opponent_model.items():
            state=torch.from_numpy(np.concatenate(self.memory_buffer[name][0],axis=0)).float()
            action=F.one_hot( torch.from_numpy( np.concatenate(self.memory_buffer[name][1],axis=0,dtype = np.int64) ), 6 ).float()
            load(model, self.save_dir+f'/{name}_in_{self.player.my_player_name}.pth')
            for _ in range(self.moa_update_time):
                for index in BatchSampler(SubsetRandomSampler(range(self.memory_buffer_count[name])), self.moa_batch_size, False):
                    action_dist=model.flatten_state_forward(state[index])
                    loss=-torch.sum(action[index]*torch.log(action_dist+1e-30),dim=1).mean()
                    self.player.opponent_optimizer[name].zero_grad()
                    loss.backward()
                    self.player.opponent_optimizer[name].step()
            torch.save(model.state_dict(),self.save_dir+f'/{name}_in_{self.player.my_player_name}.pth')
        
        self.memory_buffer={name: [[],[]] for name in self.player.opponent_model_name_list}
        self.memory_buffer_count={name:0 for name in self.player.opponent_model_name_list}
        self.player.update()
        
