import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from time import sleep

def get_ACR(filename,reward_name):
    with open(filename, encoding="utf-8-sig", mode="r") as f:
        reader=csv.DictReader(f)

        row=next(reader)
        episodes_this_iter=eval(row['episodes_this_iter'])
        reward_this_iter=eval(row[f'hist_stats/{reward_name}_reward'])

        num_of_agents=int(len(reward_this_iter)/episodes_this_iter)
        sum_reward_list=[[0] for _ in range(num_of_agents)]
        average_reward_list=[[0] for _ in range(num_of_agents)]

        for i in range(len(reward_this_iter)):
            id=i % num_of_agents
            sum_reward_list[id].append(sum_reward_list[id][-1]+reward_this_iter[i])
            average_reward_list[id].append(sum_reward_list[id][-1]/(len(sum_reward_list[id])-1))

        for row in reader:
            episodes_this_iter=eval(row['episodes_this_iter'])
            reward_this_iter=eval(row[f'hist_stats/{reward_name}_reward'])
            reward_this_iter=reward_this_iter[-episodes_this_iter*num_of_agents:]
            for i in range(len(reward_this_iter)):
                id=i % num_of_agents
                sum_reward_list[id].append(sum_reward_list[id][-1]+reward_this_iter[i])
                average_reward_list[id].append(sum_reward_list[id][-1]/(len(sum_reward_list[id])-1))

    return average_reward_list

def get_many_ACR_of_one_policy(filename,policy_name,count):
    average_reward_list=[]
    for i in range(count):
        average_reward_list.extend(get_ACR(filename,f'policy_{policy_name}{i+1}'))
    return average_reward_list

def get_DataFrame(filename,policy_name,count,class_name):
    average_reward_list=get_many_ACR_of_one_policy(filename,policy_name,count)
    list_len=len(average_reward_list)
    episode_len=len(average_reward_list[0])-1
    total_list=[]
    for lst in average_reward_list:
        lst.pop(0)
        total_list.extend(lst)
    data={'episode':list(range(1,episode_len+1))*list_len,
    'policy':[f'{class_name}']*(episode_len*list_len),
    'avg_reward':total_list}
    return pd.DataFrame(data)


concat_list=[pd.DataFrame(get_DataFrame('progress_ToMmultiMCTS.csv','ToM',4,'ToM_multiMCTS_num_based')),
pd.DataFrame(get_DataFrame('progress_multiMCTS.csv','ToM',4,'multiMCTS')),
pd.DataFrame(get_DataFrame('progress_ToMMCTS.csv','ToM',4,'ToM_MCTS')),
pd.DataFrame(get_DataFrame('progress.csv','ToM',4,'ToM_multiMCTS_type_based'))
]
frame=pd.concat(concat_list,axis=0)
frame.index=list(range(frame.shape[0]))

frame.to_csv('result/data/avg100_reward.csv')

#frame=pd.read_csv('result/data/avg_reward.csv')
sns.lineplot(x='episode',y='avg_reward',hue='policy',data=frame)

'''
data1={'ToM_multiMCTS':get_ACR('progress_ToMmultiMCTS.csv','episode')[0]}
data2={'multiMCTS':get_ACR('progress_multiMCTS.csv','episode')[0]}
frame=pd.concat([pd.DataFrame(data1),pd.DataFrame(data2)],axis=1)
sns.lineplot(data=[frame['ToM_multiMCTS'][1:],frame['multiMCTS'][1:]])
'''
'''
plt.xlabel('episodes')
plt.ylabel('average episodes reward')
'''
plt.show()