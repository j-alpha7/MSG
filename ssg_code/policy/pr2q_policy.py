from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AgentID, TrainerConfigDict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ray.rllib.policy import TorchPolicy
import itertools
from torch.distributions.categorical import Categorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.trainer import Trainer, COMMON_CONFIG
from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def generate_one_hot_sample(action_dim, repeat_time):
    tensor_list = list(itertools.product(range(action_dim), repeat=repeat_time))
    tensor_list = torch.tensor(tensor_list)
    one_hot_tensor = F.one_hot(tensor_list, action_dim)
    one_hot_tensor = one_hot_tensor.reshape(one_hot_tensor.shape[0], -1)
    return one_hot_tensor

class QFunction(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name) -> None:
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.lstm_state_size=model_config['custom_model_config']['lstm_state_size']
        self.input_channels=model_config['custom_model_config']['input_channels']
        self.world_height=model_config['custom_model_config']['world_height']
        self.world_width=model_config['custom_model_config']['world_width']
        self.player_num = 4
        self.action_dim = action_space.n
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self._preprocess=nn.Sequential(
            nn.Conv2d(self.input_channels,16,3,1,1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,1,1),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)
        self.joint_value_head = nn.Sequential(
            nn.Linear(32*self.world_height*self.world_width + self.action_dim * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        ).to(self.device)
        self.value_head = nn.Sequential(
            nn.Linear(32*self.world_height*self.world_width, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        ).to(self.device)
        self.sample = generate_one_hot_sample(self.action_dim, self.player_num - 1).to(self.device)
    
    def conv_process(self, obs):
        obs_conv = obs.reshape(obs.shape[0], self.input_channels, self.world_height, self.world_width)
        return self._preprocess(obs_conv)
    
    def cal_joint_value(self, flattened):
        return self.joint_value_head(flattened)
    
    def cal_own_value(self, flattened):
        return self.value_head(flattened)
    
    def cal_action_value(self, obs_batch):
        obs_flatten = obs_batch[:, self.action_dim:].to(self.device)
        action_mask = obs_batch[:, :self.action_dim].to(self.device)
        conved_obs_ori = self.conv_process(obs_flatten)
        conved_obs = conved_obs_ori.unsqueeze(1).repeat(1, self.action_dim**(self.player_num-1), 1)
        sample = self.sample.unsqueeze(0).repeat(obs_flatten.shape[0], 1, 1)
        s_a_j = torch.cat((conved_obs, sample), dim=-1)
        joint_value = self.cal_joint_value(s_a_j)
        self_value = self.cal_own_value(conved_obs_ori)
        opp_model = joint_value.permute(1,0,2) - self_value
        opp_model = torch.softmax(opp_model, dim = 0)
        opp_model = opp_model.permute(1,0,2)
        action_value = joint_value * opp_model
        action_value = torch.sum(action_value, dim = 1)
        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        logits = action_value + inf_mask
        return logits


class PR2QPolicy(TorchPolicy):
    def __init__(self, observation_space, action_space, config) -> None:
        super().__init__(observation_space, action_space, config)
        self.target_model = QFunction(observation_space, action_space, action_space.n, config['model'], 'target')
        self.target_model.load_state_dict(self.model.state_dict())
        # self.qf = QFunction(observation_space, action_space, action_space.n, config['model'])
        # self.joint_qf = QFunction(observation_space, action_space, action_space.n, config['model'])
        # self.joint_qf_target = QFunction(observation_space, action_space, action_space.n, config['PR2']['model_config'])
        self.action_dim = action_space.n
        self.player_num = config['env_config']['player_num']
        self.gamma = config['gamma']
        self.minibatch_size = config['sgd_minibatch_size']
        self.tau = config['tau']
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        

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
    
    def compute_actions_from_input_dict( 
            self, input_dict, explore=None, timestep=None, episodes=None, state_batches=None, **kwargs
    ):
        with torch.no_grad():
            obs_batch = torch.from_numpy(input_dict['obs'])
            action_value = self.model.cal_action_value(obs_batch).to(self.device)
            dist = Categorical(logits=action_value)
            action_batch = dist.sample()
            # for i, ep in enumerate(episodes):
            #     if ep.length == 0:
            #         ep.user_data['Vs'] = [torch.max(logits[i]).item()]
            #     else:
            #         ep.user_data['Vs'].append(torch.max(logits[i]).item())
        return action_batch.cpu().numpy(), [], {}
    
    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches, episode):
        action = []
        for other_agent in other_agent_batches.values():
            action.append(other_agent[1][SampleBatch.ACTIONS].copy())
        action = np.stack(action).T
        sample_batch['opponent_actions'] = action
        return sample_batch
    
    @override(TorchPolicy)
    def learn_on_batch(self, postprocessed_batch:SampleBatch):
        if self.model:
            self.model.train()
        train_batch = self._lazy_tensor_dict(postprocessed_batch)

        obs_batch = train_batch[SampleBatch.OBS]
        new_obs_batch = train_batch[SampleBatch.NEXT_OBS]
        actions = train_batch[SampleBatch.ACTIONS]
        opponent_actions = train_batch['opponent_actions']
        rewards = train_batch[SampleBatch.REWARDS]
        dones = train_batch[SampleBatch.DONES]

        # new_obs_batch = torch.from_numpy(postprocessed_batch[SampleBatch.NEXT_OBS].copy()).to(self.device)
        # actions = torch.from_numpy(postprocessed_batch[SampleBatch.ACTIONS].copy()).to(self.device, torch.int64)
        # opponent_actions = torch.from_numpy(postprocessed_batch['opponent_actions'].copy()).to(self.device, torch.int64)
        # rewards = torch.from_numpy(postprocessed_batch[SampleBatch.REWARDS].copy()).to(self.device)

        
        for index in BatchSampler(SubsetRandomSampler(range(obs_batch.shape[0])), self.minibatch_size, False):
            with torch.no_grad():
                # new_obs_action_value = self.target_model.cal_action_value(new_obs_batch[index]).to(self.device)
                # new_obs_value = new_obs_action_value.max(dim = 1).values
                # obs_target = rewards[index] + self.gamma * new_obs_value
                new_obs_action_value = self.target_model.cal_action_value(new_obs_batch[index]).to(self.device)
                max_action = self.model.cal_action_value(new_obs_batch[index]).to(self.device)
                max_action = max_action.max(dim = 1).indices
                new_obs_value = new_obs_action_value[torch.arange(new_obs_action_value.shape[0]), max_action]
                obs_target = rewards[index] + (1-dones[index]) * self.gamma * new_obs_value
        
            obs_flatten = obs_batch[index][:, self.action_dim:]
            conved_obs_ori = self.model.conv_process(obs_flatten).to(self.device)
            opponent_actions_onehot = F.one_hot(torch.tensor(opponent_actions[index], dtype = torch.long), self.action_dim).reshape(len(index), -1)
            s_a_j = torch.cat((conved_obs_ori, opponent_actions_onehot), dim=-1)
            joint_value = self.model.cal_joint_value(s_a_j).to(self.device)
            self_value = self.model.cal_own_value(conved_obs_ori).to(self.device)

            loss1 = F.mse_loss(joint_value[torch.arange(joint_value.shape[0]), actions[index]], obs_target)
            loss2 = F.mse_loss(self_value[torch.arange(joint_value.shape[0]), actions[index]], obs_target)
            
            self._optimizers[0].zero_grad()
            (loss1+loss2).backward()
            self._optimizers[0].step()
        
        for param1, param2 in zip(self.model.parameters(), self.target_model.parameters()):
            param2.data.copy_(self.tau * param1.data + (1 - self.tau) * param2.data)


        return {LEARNER_STATS_KEY: {}}


class PR2QTrainer(Trainer):
    _allow_unknown_configs = True

    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return COMMON_CONFIG
    
    @override(Trainer)
    def get_default_policy_class(self, config):
        return PR2QPolicy
        
