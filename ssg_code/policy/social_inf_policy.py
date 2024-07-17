import sys
sys.path.append('..')
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, validate_config,  execution_plan
from ray.rllib.utils.annotations import override
from ray.rllib.agents.trainer_template import build_trainer
import torch
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
import torch.nn.functional as F
from ray.rllib.models.modelv2 import ModelV2
from typing import Dict, List, Type, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.typing import TensorType

class SocialInfluencePolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.player_num = config['env_config']['player_num']
        self.beta = config['social_inf']['beta']
        self.moa_loss_weight = config['social_inf']['moa_loss_weight']
        self.id = config['social_inf']['id']
        self.player_name_list = [f'player_{i+1}' for i in range(self.player_num)]
        self.my_name = self.player_name_list[self.id-1]
        self.opponent_name_list = self.player_name_list.copy()
        self.opponent_name_list.pop(self.id-1)
        self.record_batch = {}
        super().__init__(observation_space, action_space, config)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        if other_agent_batches is None:
            print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
            return compute_gae_for_sample_batch(self, sample_batch,
                                                other_agent_batches, episode)

        state = sample_batch[SampleBatch.OBS]
        my_action = sample_batch[SampleBatch.ACTIONS]
        my_alive_time = my_action.shape[0]
        pad_and_truncate_action = np.zeros((self.player_num - 1, my_alive_time)) + self.action_space.n
        alive_time = []

        for i, player in enumerate(self.opponent_name_list):
            limit = min(other_agent_batches[player][1][SampleBatch.ACTIONS].shape[0], my_alive_time)
            pad_and_truncate_action[i, 0:limit] = other_agent_batches[player][1][SampleBatch.ACTIONS][0:limit]
            alive_time.append(limit)

        all_actions = np.concatenate( (pad_and_truncate_action, my_action[None, : ]), axis = 0)
        all_actions = np.transpose(all_actions)

        for i, player in enumerate(self.opponent_name_list):
            pad_len = my_alive_time - alive_time[i] + 1
            obs_len = other_agent_batches[player][1][SampleBatch.OBS].shape[1]
            sample_batch[f'pre_state_{self.id}'] = state[0:my_alive_time]
            sample_batch[f'action_condition_{player}_in_{self.id}'] = all_actions
            sample_batch[f'true_action_{player}_in_{self.id}'] = np.concatenate(
                (other_agent_batches[player][1][SampleBatch.ACTIONS][1:alive_time[i]],
                -np.ones((pad_len,), dtype=np.int32)), axis=0)
        with torch.no_grad():
            intrinsic_reward  = self.model.compute_intrinsic_reward(
                state,
                all_actions,
                my_action,
                alive_time)
            sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS].astype(np.float64)
            sample_batch[SampleBatch.REWARDS] += self.beta * intrinsic_reward

            return compute_gae_for_sample_batch(self, sample_batch,
                                                other_agent_batches, episode)

    @override(PPOTorchPolicy)
    def loss(self, model: ModelV2, dist_class: Type[ActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        moa_loss = 0
        original_loss = super().loss(model, dist_class, train_batch)
        if f'pre_state_{self.id}' not in train_batch:
            return original_loss
        conv_processed = model.conv_process(train_batch[f'pre_state_{self.id}'])
        for i, player in enumerate(self.opponent_name_list):
            avail_index = (train_batch[f'true_action_{player}_in_{self.id}'] != -1)
            action_dist = model.compute_cond_prob(conv_processed[avail_index, :],
                train_batch[f'action_condition_{player}_in_{self.id}'][avail_index,:], i)
            true_action = F.one_hot( train_batch[f'true_action_{player}_in_{self.id}'][avail_index].to(torch.int64), 6 ).float()
            moa_loss += ( -torch.sum(true_action*torch.log(action_dist+1e-30),dim=1).mean().cpu() )
        #print(original_loss)
        #print(moa_loss)
        #print('-------------------------')
        return original_loss + self.moa_loss_weight * moa_loss

SocialInfluenceTrainer = build_trainer(
    name="SocialInfluence",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=SocialInfluencePolicy,
    execution_plan=execution_plan,
)
