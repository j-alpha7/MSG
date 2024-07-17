from env import SnowDriftEnv
from model.model import RLModel, MyModel
from policy.fixed_policy import RandomPolicy, NearestDriftPolicy, FixPolicy
from policy.LOLA_policy import LOLAPolicy

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from algorithm.Alpha_Zero_MOA_relative import AlphaZeroTrainer, DEFAULT_CONFIG, AlphaZeroPolicyWrapperClass
from ray.tune.registry import ENV_CREATOR, _global_registry

register_env("Snowdrift", lambda config: SnowDriftEnv(config))
ModelCatalog.register_custom_model("MyModel",MyModel)
ModelCatalog.register_custom_model("RLModel",RLModel)

#Trainer
import ray
import ray.rllib.agents.a3c as a3c
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from gym.spaces import Box,Discrete,Dict,Space
import numpy as np
import multiprocessing as mp
import copy
import os
import time
import json
import torch

if __name__=='__main__':
    ray.init(num_gpus=0)
    config=DEFAULT_CONFIG.copy()
    config['env']='Snowdrift'
    config['env_config']={
        "player_num":4,
        "world_height":8,
        "world_width":8,
        'drift_num':6,
        'reward':6,
        'cost':4,
        "final_time":50,
        'render':False,
        'max_block_comp':0,
        'prosocial':False
        }
    env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    config_dict={
        'input_channels': config['env_config']["player_num"]+2,
        'world_height': config['env_config']['world_height'],
        'world_width': config['env_config']['world_width'],
    }
    observation_space=Dict(dict(observation=Box(low=0,high=1,shape=(config['env_config']["player_num"]+2,config['env_config']["world_height"],config['env_config']["world_width"]),dtype=np.int8),
                                action_mask=Box(low=0,high=1,shape=(6,),dtype=np.int8)
                                ))
                                     
    action_space=Discrete(6)#{UP,DOWN,LEFT,RIGHT,STAY,LINK}

    config['_disable_preprocessor_api']=False
    config['create_env_on_driver']=True
    config['framework']='torch'
    config['num_gpus']=0
    config['num_workers']=1
    config['num_envs_per_worker']=1
    config['rollout_fragment_length']=200
    config['gamma']=0.95
    config['train_batch_size']=200
    config['sgd_minibatch_size']=128
    config['num_sgd_iter']=1
    config['lr']=5e-4

    config['multiagent']['policies_to_train']=['ToM1','ToM2','ToM3','ToM4','lola1','lola2','lola3','lola4']
    config['model']['custom_model']='MyModel'
    config['model']['custom_model_config']=config_dict

    fixed_config_dict_list=[]
    for id in range(4):
        fixed_config_dict_list.append(copy.deepcopy(config_dict))
        fixed_config_dict_list[-1]['id'] = id+1

    def policy_mapping(agent_id, episode, worker, **kwargs):
        if agent_id=='player_1':
            return 'train1'
        elif agent_id=='player_2':
            return 'train2'
        elif agent_id=='player_3':
            return 'train3'
        else:
            return 'train4'
    config['multiagent']['policy_mapping_fn']=policy_mapping

    render_env_config=config['env_config'].copy()
    render_env_config['render']=True

    config['mcts_config']={
        "puct_coefficient":15,
        "num_simulations": 200,
        "temperature":1.5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": False,
        "add_dirichlet_noise": False
        }

    ToM_config={
        'discount_factor':0.98,
        'gamma':0.95,
        'my_id':1,
        'tree_num':5,
        'Q_temperature':2,
        'evaluation':False}

    ToM_config['env_config']=config['env_config'].copy()
    ToM_config['env_creator']=env_creator
    config['ToM_config']=ToM_config
    ToM_eval_config=copy.deepcopy(config['ToM_config'])
    ToM_eval_config['evaluation']=True

    config['moa_config']={
        'moa_batch_size':512,
        'moa_update_time':15,
        'moa_buffer_capacity':1000,
        'shovel_buffer_capacity':100,
        #'load_dir':None,
        #'load_dir':'./params/exp20230410_004313_4p', # no-long-ToM
        #'load_dir':'./params/exp20230423_012756_4p',
        'load_dir':'./params/exp20230119_004042_4p',
        #'load_dir':'./params/exp20230118_165322_4p',
        #'load_dir':'./params/exp20230116_235849_4p',
        #'load_dir':'./params/exp20230116_005851_4p',
        #'load_dir':'./params/exp20230114_222914_4p',
        #'load_dir':'./params/exp20230113_122853_4p',
        #'load_dir':'./params/exp20221019_162421_4p',
        'save_dir':f'./params/exp{time.strftime("%Y%m%d_%H%M%S")}_{config["env_config"]["player_num"]}p'
    }
    os.makedirs(config['moa_config']['save_dir'])

    config_list=[]
    for id in range(4):
        config_list.append(copy.deepcopy(config))
        config_list[-1]['ToM_config']['my_id']=id+1

    '''
    config["evaluation_interval"]=1
    config["evaluation_num_episodes"]=3
    config['evaluation_config']={
        'render_env':True,
        'env_config':render_env_config,
        'ToM_config':ToM_eval_config
        }
    '''
    config['lola_config']={
        'moa_batch_size':128,
        'moa_update_time':15,
        'buffer_capacity':200,
        'lr':2e-4,
        'gamma': 0.99,
        'id': 1,
        'env_creator':env_creator,
        'env_num':25,
        'load_dir':None,
        #'load_dir':'./params/lola/exp20230123_115158_4p',
        'load_dir':'./params/lola/exp20230123_185154_4p',
        #'load_dir':'./params/lola/exp20230123_173827_4p',
        #'load_dir':'./params/lola/exp20221203_125305_4p',
        'save_dir':f'./params/lola/exp{time.strftime("%Y%m%d_%H%M%S")}_{config["env_config"]["player_num"]}p',
        'train':False,
    }
    os.makedirs(config['lola_config']['save_dir'])

    lola_config=[]
    for id in range(4):
        lola_config.append(copy.deepcopy(config))
        lola_config[-1]['lola_config']['id']=id+1

    a3c_config=a3c.DEFAULT_CONFIG.copy()
    a3c_config['env']=config['env']
    a3c_config['env_config']=config['env_config']
    a3c_config['_disable_preprocessor_api']=config['_disable_preprocessor_api']
    a3c_config['create_env_on_driver']=config['create_env_on_driver']
    a3c_config['framework']=config['framework']
    a3c_config['num_gpus']=1
    a3c_config['num_workers']=8
    a3c_config['num_envs_per_worker']=5
    a3c_config['rollout_fragment_length']=100
    a3c_config['gamma']=0.99
    a3c_config['train_batch_size']=4000
    a3c_config['multiagent']['policies_to_train']=['train1','train2','train3','train4',]# 'prosocial1', 'prosocial2', 'prosocial3', 'prosocial4']
    a3c_config['model']['custom_model']='RLModel'
    a3c_config['model']['custom_model_config']=config_dict
    a3c_config['model']['custom_model_config']['lstm_state_size']=256
    
    a3c_config['multiagent']['policy_mapping_fn']=policy_mapping
    '''
    a3c_config["evaluation_interval"]=1
    a3c_config["evaluation_num_episodes"]=5
    a3c_config['evaluation_config']={
        'render_env':True,
        'env_config':render_env_config
    }
    '''
    
    policies={
        'train1':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'train2':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'train3':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'train4':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'prosocial1':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'prosocial2':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'prosocial3':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'prosocial4':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'random':PolicySpec(RandomPolicy,observation_space,action_space),
        'fix1':PolicySpec(FixPolicy, observation_space, action_space, fixed_config_dict_list[0]),
        'fix2':PolicySpec(FixPolicy, observation_space, action_space, fixed_config_dict_list[1]),
        'fix3':PolicySpec(FixPolicy, observation_space, action_space, fixed_config_dict_list[2]),
        'fix4':PolicySpec(FixPolicy, observation_space, action_space, fixed_config_dict_list[3]),
        'cleaner1':PolicySpec(NearestDriftPolicy, observation_space, action_space, fixed_config_dict_list[0]),
        'cleaner2':PolicySpec(NearestDriftPolicy, observation_space, action_space, fixed_config_dict_list[1]),
        'cleaner3':PolicySpec(NearestDriftPolicy, observation_space, action_space, fixed_config_dict_list[2]),
        'cleaner4':PolicySpec(NearestDriftPolicy, observation_space, action_space, fixed_config_dict_list[3]),
        'lola1':PolicySpec(LOLAPolicy,observation_space,action_space,lola_config[0]),
        'lola2':PolicySpec(LOLAPolicy,observation_space,action_space,lola_config[1]),
        'lola3':PolicySpec(LOLAPolicy,observation_space,action_space,lola_config[2]),
        'lola4':PolicySpec(LOLAPolicy,observation_space,action_space,lola_config[3]),
        'social1':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'social2':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'social3':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'social4':PolicySpec(A3CTorchPolicy,observation_space,action_space,a3c_config),
        'ToM1':PolicySpec(AlphaZeroPolicyWrapperClass,
        observation_space=observation_space,
        action_space=action_space,
        config=config_list[0]),
        'ToM2':PolicySpec(AlphaZeroPolicyWrapperClass,
        observation_space=observation_space,
        action_space=action_space,
        config=config_list[1]),
        'ToM3':PolicySpec(AlphaZeroPolicyWrapperClass,
        observation_space=observation_space,
        action_space=action_space,
        config=config_list[2]),
        'ToM4':PolicySpec(AlphaZeroPolicyWrapperClass,
        observation_space=observation_space,
        action_space=action_space,
        config=config_list[3]),
    }

    a3c_config['multiagent']['policies']=policies
    config['multiagent']['policies']=policies
    '''
    json_config=copy.deepcopy(config)
    json_config['multiagent']=None
    json_config.pop('callbacks')
    json_config.pop('sample_collector')
    json_config['ToM_config']['env_creator']=None
    with open(config['moa_config']['save_dir']+'/config.json','a') as f:
        json.dump(json_config,f)
    '''
    # mp.set_start_method('forkserver')
    # tom_trainer=AlphaZeroTrainer(config)
    #tom_trainer.restore('/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-04-10_00-43-13tmer3nx9/checkpoint_0000101/checkpoint-101') no-long-ToM
    #tom_trainer.restore('/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-19_00-40-42tlhif_28/checkpoint_000081/checkpoint-81')
    #tom_trainer.restore('/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-18_16-53-22zibw3hpf/checkpoint_000082/checkpoint-82')
    #tom_trainer.restore('/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-16_23-58-495pl5389l/checkpoint_000061/checkpoint-61')
    #tom_trainer.restore('/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-17_23-28-39mu6z723r/checkpoint_000041/checkpoint-41')
    #tom_trainer.restore('/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-16_00-58-51o9lt370d/checkpoint_000223/checkpoint-223')
    #tom_trainer.restore('/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-14_22-29-14dhyqv2z4/checkpoint_000162/checkpoint-162')
    #tom_trainer.restore('/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-13_12-28-53zaat2esu/checkpoint_000081/checkpoint-81')
    
    a3c_trainer=a3c.A3CTrainer(a3c_config)
    #tom_trainer.restore('/root/ray_results/AlphaZeroTrainer_Snowdrift_2022-10-19_16-24-212c1x8o34/checkpoint_000271/checkpoint-271')
    #/root/ray_results/A3C_Snowdrift_2022-10-18_06-15-545zt1a13g/checkpoint_005001/checkpoint-5001
    #prosocial: /root/ray_results/A3C_Snowdrift_2022-10-18_06-17-31x5rupa8o/checkpoint_005001/checkpoint-5001
    #3RL1rand: AlphaZeroTrainer_Snowdrift_2022-10-27_11-47-507wvk3dij
    #1RL3rand: AlphaZeroTrainer_Snowdrift_2022-10-27_11-50-09q706x11r
    #4rand:/root/ray_results/AlphaZeroTrainer_Snowdrift_2022-10-27_11-52-317vom5enh/checkpoint_005002/checkpoint-5002
    #1ToM3rand:AlphaZeroTrainer_Snowdrift_2022-10-27_05-17-02ev8sffpz

    #3fix1random:A3C_Snowdrift_2022-10-27_16-24-414qtoo_et
    #1fix3random:A3C_Snowdrift_2022-10-27_16-27-52zhdfrhax
    #1fix3train:A3C_Snowdrift_2022-10-27_16-31-366kib10uv
    #3fix1train:/root/ray_results/A3C_Snowdrift_2022-10-27_16-35-394l26nirf/checkpoint_005002/checkpoint-5002
    #tom_trainer.restore('/root/ray_results/A3C_Snowdrift_2022-10-27_02-08-35h73h218w/checkpoint_005001/checkpoint-5001')
    #a3c_trainer.restore('./checkpoint_009002/checkpoint-9002')
    a3c_trainer.restore('./checkpoint_010004/checkpoint-10004')
    #tom_trainer.restore('/root/ray_results/A3C_Snowdrift_2023-01-20_00-07-064keeooxt/checkpoint_009003/checkpoint-9003')
    #4prosocial:/root/ray_results/A3C_Snowdrift_2022-10-27_16-49-39aeidmz_a/checkpoint_009002/checkpoint-9002
    #3prosocial+1RL:  /root/ray_results/A3C_Snowdrift_2022-10-28_02-34-369trpd9vh/checkpoint_009003/checkpoint-9003
    #1prosocial+3RL:A3C_Snowdrift_2022-10-28_02-37-17b2fk1_1n
    #3prosocial+1fixed:/root/ray_results/A3C_Snowdrift_2022-10-28_02-43-54m9f8cg7v/checkpoint_009003/checkpoint-9003
    #3prosocial+1random:A3C_Snowdrift_2022-10-28_02-41-03t25yzf19
    #1prosocial+3fixed:/root/ray_results/A3C_Snowdrift_2022-10-28_02-46-23iax765me/checkpoint_009003/checkpoint-9003
    #1prosocial+3random:/root/ray_results/A3C_Snowdrift_2022-10-28_02-50-12qfdnua10/checkpoint_009003/checkpoint-9003
    
    #1ToM+3cleaner:/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-20_20-24-25j1bugegz/checkpoint_000102/checkpoint-102
    #1ToM+3random:/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-20_23-35-403pzlhnmb
    #3ToM+1cleaner:/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-21_00-45-5503ozkts0/checkpoint_000102/checkpoint-102
    #3ToM+1random:/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-21_00-43-36w_f5yc8g/checkpoint_000102/checkpoint-102
    #1ToM+3prosocial:/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-21_14-43-087tdwmqif/checkpoint_009024/checkpoint-9024
    #1ToM+3train:/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-21_15-44-48opk1dson/checkpoint_009024/checkpoint-9024
    #3ToM+1prosocial:/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-21_19-59-25_hh1v1r_
    #3ToM+1train:/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-21_19-59-554q1n0t4t
    
    #1LOLA+3random:/root/ray_results/A3C_Snowdrift_2023-01-24_10-05-165e8oadhr/checkpoint_009004/checkpoint-9004
    #1LOLA+3cleaner:/root/ray_results/A3C_Snowdrift_2023-01-24_10-13-11y94rcrm3/checkpoint_009004/checkpoint-9004
    #1LOLA+3train:/root/ray_results/A3C_Snowdrift_2023-01-24_10-12-40lwx1adzs/checkpoint_009024/checkpoint-9024
    #1LOLA+3prosocial:/root/ray_results/A3C_Snowdrift_2023-01-24_10-16-09f_2eghug/checkpoint_009024/checkpoint-9024
    #3LOLA+1cleaner:/root/ray_results/A3C_Snowdrift_2023-01-24_10-53-01ccatbnqq/checkpoint_009004/checkpoint-9004
    #3LOLA+1random: /root/ray_results/A3C_Snowdrift_2023-01-24_10-50-59kxsxa9r1/checkpoint_009004/checkpoint-9004
    #3LOLA+1train:/root/ray_results/A3C_Snowdrift_2023-01-24_11-03-09ew3t5976/checkpoint_009024/checkpoint-9024
    #3LOLA+1prosocial:/root/ray_results/A3C_Snowdrift_2023-01-24_11-03-59nnt97ey8/checkpoint_009024/checkpoint-9024
    #3LOLA+1ToM:/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-24_12-48-36zgqbr9m7/checkpoint_000021/checkpoint-21
    #3train+1ToM:/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-24_13-13-25tns7kivp/checkpoint_009024/checkpoint-9024
    #1LOLA+3ToM:/root/ray_results/AlphaZeroTrainer_Snowdrift_2023-01-24_13-22-226ikhheow/checkpoint_009044/checkpoint-9044
    '''
    for i in range(4):
        tom_trainer.get_policy(f'ToM{i+1}').set_weights(torch.load(f'./state_dict/ToM/ToM{i+1}.pth'))
    '''
    for i in range(5001):
        result=a3c_trainer.train()
        print(pretty_print(result))
        '''
        for _ in range(4):
            torch.save(tom_trainer.get_policy(f'ToM{i+1}').get_weights(),f'./state_dict/ToM/ToM{i+1}.pth')
        print('AAA')
        '''
        if i%1000==0:
            checkpoint = a3c_trainer.save()
            #checkpoint_a3c=a3c_trainer.save()
            print("checkpoint saved at", checkpoint)
        

