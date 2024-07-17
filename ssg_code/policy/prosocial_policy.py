import sys
sys.path.append('..')
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np

class ProsocialPolicy(A3CTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.player_num=config['env_config']['player_num']
        self.id = config['id']
        self.player_name_list = [f'player_{i+1}' for i in range(self.player_num)]
        self.my_name = self.player_name_list[self.id-1]
        self.opponent_name_list = self.player_name_list.copy()
        self.opponent_name_list.pop(self.id-1)
        super().__init__(observation_space, action_space, config)

    @override(A3CTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        if "original_reward" not in sample_batch:
            sample_batch['original_reward'] = sample_batch[SampleBatch.REWARDS].copy()
        reward = sample_batch['original_reward']
        length = reward.shape[0]
        sum_reward =reward.copy()
        if other_agent_batches is not None:
            for player in self.opponent_name_list:
                batch = other_agent_batches[player][1]
                pad_reward = np.zeros_like(reward)
                if "original_reward" not in batch:
                    batch["original_reward"] = batch[SampleBatch.REWARDS].copy()
                other_reward = batch["original_reward"]
                # print(other_reward)
                # print(player)
                # print(self.id)
                # print('=======================')
                ind = min(length, other_reward.shape[0])
                pad_reward[:ind] = other_reward[:ind]
                sum_reward += pad_reward
        mean_reward = sum_reward.astype(np.float32) / self.player_num
        sample_batch[SampleBatch.REWARDS] = mean_reward

        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)
