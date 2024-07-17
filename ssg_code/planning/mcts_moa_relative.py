"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import math
from ssl import DefaultVerifyPaths
import numpy as np
import copy
from time import sleep, time


class Node:
    def __init__(self, action, obs, done, reward, state, model_list, model_id_list, target_prey, drift_num, drift_pos, mcts, id, count, gamma, parent=None):
        self.env = parent.env
        self.action = action  # Action used to go to this state

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.action_space_size = self.env.action_space.n
        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32)  # Q
        self.child_priors = np.zeros(
            [self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32)  # N

        self.reward = reward
        self.done = done
        self.state = state
        self.obs = obs

        self.model_list=model_list
        self.model_id_list=model_id_list
        self.player_name=[f'player_{id+1}' for id in self.model_id_list]
        self.target_prey=target_prey
        self.drift_num=drift_num
        self.drift_pos=drift_pos.copy()

        self.mcts = mcts

        self.id=id
        self.count=count

        self.gamma=gamma

        self.valid_actions = obs["action_mask"].astype(bool)
        self.action_dict=None
        #self.valid_actions = np.array([1,1,1,1,1,1]).astype(bool)
        #self.valid_actions = self.get_action_mask()

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        # TODO (weak todo) add "softmax" version of the Q-value
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * self.child_priors / (
            1 + self.child_number_visits)

    def best_action(self):
        """
        :return: action
        """
        
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        masked_child_score[~self.valid_actions] = -np.inf
        
        '''
        normalization = np.max(np.abs(self.child_Q()))
        if normalization<1:
            child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        else:
            child_score = self.child_Q()/normalization + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        masked_child_score[self.child_number_visits==0] = np.inf
        masked_child_score[~self.valid_actions] = -np.inf
        '''
        '''
        child_score = np.sign(self.child_Q())*0.4*np.log(1+np.abs(self.child_Q())) + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        masked_child_score[self.child_number_visits==0] = np.inf
        masked_child_score[~self.valid_actions] = -np.inf
        '''
        return np.argmax(masked_child_score)

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        if action not in self.children:
            self.env.set_state(self.state)
            if self.action_dict == None:
                self.action_dict = {}
                avail_drift=self.get_avail_drift()
                all_obs=self.env.__obs__(list(range(len(self.model_id_list)+1)))
                player_pos=self.state['player_pos']
                for i,player in enumerate(self.player_name):
                    model_id=self.model_id_list[i]
                    target=self.get_target(i,avail_drift)
                    if target == self.drift_num:
                        self.action_dict[player]=np.random.choice(np.flatnonzero(all_obs[f'player_{model_id+1}']['action_mask'][:5]))
                    else:
                        action_prob_dist=self.model_list[player].get_action(all_obs[f'player_{model_id+1}'],
                            self.state['time'], self.drift_pos[target]-player_pos[model_id])
                        self.action_dict[player]=np.random.choice(6,p=action_prob_dist)
            
            self.action_dict[f'player_{self.id}']=action
            obs_all, reward_all, done_all, _ = self.env.step(self.action_dict)
            done=done_all[f'player_{self.id}']
            obs=obs_all[f'player_{self.id}']
            reward=reward_all[f'player_{self.id}']
            next_state = copy.deepcopy(self.env.get_state())
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                model_list=self.model_list,
                model_id_list=self.model_id_list,
                target_prey=self.target_prey,
                drift_num=self.drift_num,
                drift_pos=self.drift_pos,
                obs=obs,
                mcts=self.mcts,
                id=self.id,
                count=self.count+1,
                gamma=self.gamma)
        return self.children[action]

    def backup(self, value):
        current = self
        curr_value = value + self.reward
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += curr_value
            current = current.parent
            curr_value = self.gamma*curr_value + current.reward
    
    def get_target(self, i, avail_drift):
        for target in self.target_prey[i]:
            if target in avail_drift:
                return target
    
    def get_avail_drift(self):
        obs=self.state['state']
        avail_drift=[self.drift_num]
        for pos in self.drift_pos:
            if pos[0]!=-1 and obs[-1,pos[0],pos[1]]==0:
                pos[0]=pos[1]=1

        for i,pos in enumerate(self.drift_pos):
            if pos[0]!=-1:
                avail_drift.append(i)
        return avail_drift

class RootParentNode:
    def __init__(self, env):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.env = env
        self.reward = 0


class MCTS:
    def __init__(self, model, mcts_param, evaluation):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]
        self.evaluation=evaluation

    def compute_action(self, node):
        for _ in range(self.num_sims):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.model.compute_priors_and_value(leaf.obs, leaf.state['time'])
                if self.add_dirichlet_noise:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet(
                        [self.dir_noise] * child_priors.size
                    )

                leaf.expand(child_priors)
            leaf.backup(value)
            
        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        Q_value=node.child_Q()
        Q_value[~node.valid_actions] = -np.inf
        tree_policy = tree_policy / np.max(
            tree_policy)  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = np.argmax(tree_policy)
        else:
            # otherwise sample an action according to tree policy probabilities
            action = np.random.choice(
                np.arange(node.action_space_size), p=tree_policy)
        ''' 
        pos = node.state['player_pos'][node.id - 1]
        if node.state['state'][-1, pos[0], pos[1]]==1:
            print(node.child_total_value)
            print(node.child_number_visits)
            print(node.child_Q())
            print(action)
            print(node.state['time'])
            print(node.target_prey)
            
            p,v=self.model.compute_priors_and_value(node.obs)
            print(p)
            print(np.sum(np.log(p+1e-30)*tree_policy))
            print(v)
            
            
            print(node.children[0].child_total_value)
            print(node.children[0].child_number_visits)
            print(node.children[0].child_Q())
        '''
        return tree_policy, action, node.children[action],Q_value
