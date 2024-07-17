from abc import ABC
import numpy as np
import torch
import torch.nn as nn

from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
import torch.nn.functional as F


def convert_to_tensor(arr):
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor


class MyModel(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.input_channels=model_config['custom_model_config']['input_channels']
        self.world_height=model_config['custom_model_config']['world_height']
        self.world_width=model_config['custom_model_config']['world_width']
        
        if torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        self.shared_layers=nn.Sequential(
            nn.Conv2d(self.input_channels,16,3,1,1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,1,1),
            nn.Flatten()
        ).to(self.device)

        self.flatten_layers = nn.Sequential(
            nn.Linear(32*self.world_height*self.world_width+1,512),
            nn.ReLU(),
        ).to(self.device)
        self.actor_layers = nn.Linear(512,6).to(self.device)
        self.critic_layers = nn.Linear(512,1).to(self.device)

        self._value_out = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict, state, seq_lens, time):
        if 'observation' not in input_dict.keys():
            obs_flatten=input_dict['obs_flat']
            x=obs_flatten[...,6:].reshape(obs_flatten.shape[0],self.input_channels,self.world_height,self.world_width)
            action_mask=obs_flatten[...,:6]
            x=x.to(self.device)
            action_mask=action_mask.to(self.device)
        else:
            x = input_dict["observation"]
            action_mask=input_dict['action_mask']
            x=convert_to_tensor(x).to(self.device).float()
            action_mask=convert_to_tensor(action_mask).to(self.device).float()
            x=x.unsqueeze(0)
            action_mask=action_mask.unsqueeze(0)
        
        time=time.to(self.device)
        x = self.shared_layers(x)
        x = torch.cat((x,time),dim=1)
        x = self.flatten_layers(x)
        # actor outputs
        logits = self.actor_layers(x)
        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)

        # compute value
        self._value_out = self.critic_layers(x)

        return (logits+inf_mask).cpu(), []

    def value_function(self):
        return self._value_out.cpu()

    def compute_priors_and_value(self, obs, time):
        with torch.no_grad():
            model_out = self.forward(obs, None, [1], torch.tensor([[time]]))
            logits, _= model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value

class RLModel(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.lstm_state_size=model_config['custom_model_config']['lstm_state_size']
        self.input_channels=model_config['custom_model_config']['input_channels']
        self.player_num=model_config['custom_model_config']['input_channels']-8
        self.world_height=model_config['custom_model_config']['world_height']
        self.world_width=model_config['custom_model_config']['world_width']
        
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
        )

        self.lstm=nn.LSTM(32*self.world_height*self.world_width,
        self.lstm_state_size,
        batch_first=True)
        
        '''
        self._preprocess=nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.world_height*self.world_width*self.input_channels,128),
            nn.ReLU()
        )
        self.lstm=nn.LSTM(128,
        self.lstm_state_size,
        batch_first=True)
        '''
        self._action_branch=nn.Linear(self.lstm_state_size,6)
        self._value_branch=nn.Linear(self.lstm_state_size,1)
        self._features = None

    def get_initial_state(self):
        return [self._preprocess[0].weight.new(1,self.lstm_state_size).zero_().squeeze(0),
        self._preprocess[0].weight.new(1,self.lstm_state_size).zero_().squeeze(0)]

    def forward_rnn(self,inputs,state,seq_lens):
        state = [state[0], state[1]]
        obs_flatten=inputs[:,:,6:].float()
        obs=obs_flatten.reshape(obs_flatten.shape[0],obs_flatten.shape[1],self.input_channels,self.world_height,self.world_width)
        obs_postprocess_set=[]
        for i in range(obs.shape[1]):
            obs_postprocess_set.append(self._preprocess(obs[:,i,...]))
        obs_postprocessed=torch.stack(obs_postprocess_set,dim=1)

        self._features,[h,c]=self.lstm(obs_postprocessed,[torch.unsqueeze(state[0],0),torch.unsqueeze(state[1],0)])
        action_mask=inputs[:,:,:6].float()
        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        action_logits=self._action_branch(self._features)
        return action_logits+inf_mask, [torch.squeeze(h,0),torch.squeeze(c,0)]

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._value_branch(self._features),[-1])
