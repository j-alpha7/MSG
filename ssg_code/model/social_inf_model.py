from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
import torch
from torch import nn
import torch.nn.functional as F

class SocialInfluenceModel(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.lstm_state_size=model_config['custom_model_config']['lstm_state_size']
        self.input_channels=model_config['custom_model_config']['input_channels']
        self.world_height=model_config['custom_model_config']['world_height']
        self.world_width=model_config['custom_model_config']['world_width']
        self.player_num = self.input_channels - 2
        self.action_num = action_space.n

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
        self.lstm=nn.LSTM(32*self.world_height*self.world_width,
        self.lstm_state_size,
        batch_first=True).to(self.device)

        self._action_branch=nn.Linear(self.lstm_state_size, self.action_num).to(self.device)
        self._value_branch=nn.Linear(self.lstm_state_size,1).to(self.device)
        self._features = None

        self.moa_action_branch = []
        for _ in range(self.player_num - 1):
            self.moa_action_branch.append(
                nn.Sequential(
                    nn.Linear( 32*self.world_height*self.world_width + (self.action_num+1) * self.player_num, 512),
                    nn.ReLU(),
                    nn.Linear( 512, self.action_num )
                ).to(self.device)
            )

    def get_initial_state(self):
        return [self._preprocess[0].weight.new(1,self.lstm_state_size).zero_().squeeze(0),
        self._preprocess[0].weight.new(1,self.lstm_state_size).zero_().squeeze(0)]

    def forward_rnn(self,inputs,state,seq_lens):
        state = [state[0], state[1]]
        obs_flatten=inputs[:,:,self.action_num:].float()
        obs=obs_flatten.reshape(obs_flatten.shape[0],obs_flatten.shape[1],self.world_height,self.world_width,self.input_channels)
        obs=obs.permute(0,1,4,2,3)
        obs_postprocess_set=[]
        for i in range(obs.shape[1]):
            obs_postprocess_set.append(self._preprocess(obs[:,i,...]))
        self.obs_postprocessed=torch.stack(obs_postprocess_set,dim=1)
        self._features,[h,c]=self.lstm(self.obs_postprocessed,[torch.unsqueeze(state[0],0),torch.unsqueeze(state[1],0)])

        action_mask=inputs[:,:,:self.action_num].float()
        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        action_logits=self._action_branch(self._features)
        return action_logits+inf_mask, [torch.squeeze(h,0),torch.squeeze(c,0)]

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._value_branch(self._features),[-1])

    def conv_process(self, inputs):
        obs = inputs[:, self.action_num: ].to(self.device).float()
        obs = obs.reshape(inputs.shape[0], self.input_channels,self.world_height,self.world_width)
        return self._preprocess(obs)

    def compute_cond_prob(self, conv_processed, cond_action, i):
        cond_action = cond_action.to(self.device).to(torch.int64)
        cond_action = F.one_hot( cond_action, self.action_num+1)
        cond_action = cond_action.reshape(conv_processed.shape[0], (self.action_num+1)*self.player_num).float()

        x = torch.cat((conv_processed, cond_action), dim = 1)
        x = self.moa_action_branch[i](x)
        x = nn.Softmax(dim=-1)(x)
        return x

    def compute_intrinsic_reward(self, inputs, all_action, my_action, alive_time):
        obs = torch.tensor(inputs[:, self.action_num: ], device=self.device).float()
        action_mask = torch.tensor(inputs[:, :self.action_num], device=self.device).float()
        obs = obs.reshape(inputs.shape[0], self.input_channels,self.world_height,self.world_width)
        all_action = torch.tensor(all_action).to(self.device)
        #print(all_action.shape)

        obs_preprocess = self._preprocess(obs)
        obs_preprocess_time = obs_preprocess.unsqueeze(0)
        action_mask_time = action_mask.unsqueeze(0)
        state_batch = self.get_initial_state()
        for i in range(2):
            state_batch[i] = state_batch[i].unsqueeze(0)
        my_action_prob, _ = self.cal_my_action_prob(obs_preprocess_time, action_mask_time, state_batch)
        my_action_prob = my_action_prob.squeeze(0)
        #print(my_action_prob.shape)
        #print('AAA')

        cond_prob_list = [[] for _ in range(self.player_num - 1)]

        for i in range(self.action_num):
            all_action_one_hot = all_action.to(dtype=torch.int64)
            all_action_one_hot[ : , -1] = i
            all_action_one_hot = F.one_hot( all_action_one_hot, self.action_num+1 )
            all_action_one_hot = all_action_one_hot.reshape(obs.shape[0], (self.action_num+1)*self.player_num).float()
            x = torch.cat((obs_preprocess, all_action_one_hot), dim=1).float()

            for j in range(self.player_num - 1):
                y = self.moa_action_branch[j](x)
                y = nn.Softmax(dim=-1)(y)
                cond_prob_list[j].append(y)
                #print(cond_prob_list[j][-1].shape)

        marginal_prob_list = []
        for j in range(self.player_num - 1):
            marginal_prob = torch.zeros(obs_preprocess.shape[0], self.action_num)
            for i in range(self.action_num):
                '''
                print(marginal_prob.device)
                print(cond_prob_list[j][i].device)
                print(my_action_prob[:,i].device)
                print('--------------------------')
                '''
                marginal_prob += (cond_prob_list[j][i] * (my_action_prob[:,i].unsqueeze(1)))

            marginal_prob_list.append(marginal_prob)


        true_conditional_prob_list = []
        for j in range(self.player_num - 1):
            true_conditional_prob = torch.zeros(obs_preprocess.shape[0], self.action_num)
            for id, ac in enumerate(my_action):
                true_conditional_prob[id] = cond_prob_list[j][ac][id]
            true_conditional_prob_list.append(true_conditional_prob)

        intrinsic_reward = torch.zeros(obs_preprocess.shape[0])
        for j in range(self.player_num - 1):
            intrinsic_reward += torch.mean( F.kl_div(
                torch.log(marginal_prob_list[j]+1e-30),
                true_conditional_prob_list[j],
                reduction='none'
                ), dim=1)

        return intrinsic_reward.numpy()

    def cal_my_action_prob(self,obs_postprocessed,action_mask, state):
        state = [state[0], state[1]]
        self._features,[h,c]=self.lstm(obs_postprocessed,[torch.unsqueeze(state[0],0),torch.unsqueeze(state[1],0)])

        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        action_logits=self._action_branch(self._features) + inf_mask
        prob = nn.Softmax(dim=-1)(action_logits)
        return prob, [torch.squeeze(h,0),torch.squeeze(c,0)]

