import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FLOAT_MIN=-1e38
FLOAT_MAX=1e38


class MOAModel(nn.Module):
    def __init__(self, model_config):
        nn.Module.__init__(self)
        self.input_channels=model_config['input_channels']
        self.world_height=model_config['world_height']
        self.world_width=model_config['world_width']
        
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
            nn.Linear(32*self.world_height*self.world_width+3,512),
            nn.ReLU(),
        ).to(self.device)
        self.actor_layers = nn.Linear(512,6).to(self.device)
        self.critic_layers = nn.Linear(512,1).to(self.device)

        self._value_out = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, obs_flatten, time, target_layer):
        x=obs_flatten[...,6:].reshape(obs_flatten.shape[0],self.input_channels,self.world_height,self.world_width)
        action_mask=obs_flatten[...,:6]
        x=x.to(self.device)
        action_mask=action_mask.to(self.device)
        time=time.to(self.device).float().unsqueeze(1)

        x = self.shared_layers(x)
        x = torch.cat((x, time, target_layer),dim=1)
        x = self.flatten_layers(x)
        # actor outputs
        logits = self.actor_layers(x)
        inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        priors=nn.Softmax(dim=-1)(logits+inf_mask)

        return priors.cpu()

    def get_action(self, obs, time, target_pos):
        with torch.no_grad():
            x=torch.from_numpy(obs['observation']).to(self.device).float().unsqueeze(0)
            action_mask=torch.from_numpy(obs['action_mask']).to(self.device).float().unsqueeze(0)
            time=torch.tensor([[time]]).to(self.device).float()
            target_layer=torch.from_numpy(target_pos).to(self.device).float().unsqueeze(0)

            x = self.shared_layers(x)
            x = torch.cat((x, time, target_layer),dim=1)
            x = self.flatten_layers(x)
            # actor outputs
            logits = self.actor_layers(x)
            inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
            priors = nn.Softmax(dim=-1)(logits+inf_mask)

        return priors.squeeze().cpu().numpy()

    def get_action_prob(self, obs, time, action, avail_drift_layer):
        with torch.no_grad():
            x=torch.from_numpy(obs['observation']).to(self.device).float().unsqueeze(0)
            action_mask=torch.from_numpy(obs['action_mask']).to(self.device).float().unsqueeze(0)
            time=torch.tensor([[time]]).to(self.device).float()
            x = self.shared_layers(x)

            prob_list=[]
            for layer in avail_drift_layer:
                if layer is None:
                    prob_list.append(0)
                else:
                    target_layer=torch.from_numpy(layer).to(self.device).float().unsqueeze(0)
                    y = self.flatten_layers(torch.cat((x, time, target_layer),dim=1))
                    logits = self.actor_layers(y)
                    inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
                    prob_list.append( 
                        nn.Softmax(dim=-1)(logits+inf_mask).squeeze()[action].cpu().item()
                    )
            prob_list=np.array(prob_list, dtype=np.float32)
        return prob_list
