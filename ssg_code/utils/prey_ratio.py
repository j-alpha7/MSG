import csv
from itertools import count
from operator import concat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from time import sleep

def get_AR_range(filename,reward_name,_counting_range):
    counting_range=int(_counting_range)
    with open(filename, encoding="utf-8-sig", mode="r") as f:
        reader=csv.DictReader(f)

        row=next(reader)
        episodes_this_iter=eval(row['episodes_this_iter'])
        reward_this_iter=eval(row[f'hist_stats/{reward_name}_reward'])

        num_of_agents=int(len(reward_this_iter)/episodes_this_iter)
        stag_time=[0]*num_of_agents
        hare_time=[0]*num_of_agents
        blank_time=[0]*num_of_agents
        length=episodes_this_iter
            
        for row in reader:
            episodes_this_iter=eval(row['episodes_this_iter'])
            reward_this_iter=eval(row[f'hist_stats/{reward_name}_reward'])
            reward_this_iter=reward_this_iter[-episodes_this_iter*num_of_agents:]
            length+=episodes_this_iter
            if length>counting_range:
                for i in range(len(reward_this_iter)):
                    id=i % num_of_agents
                    if reward_this_iter[i]<0.1:
                        blank_time[id]+=1
                    elif reward_this_iter[i]<1.1:
                        hare_time[id]+=1
                    else:
                        stag_time[id]+=1
                    
                
    return stag_time,hare_time,blank_time

def get_many_AR_range_of_one_policy(filename,policy_name,count,counting_range):
    stag_sum,hare_sum,blank_sum=0,0,0
    for i in range(count):
        prey_tuple=get_AR_range(filename,f'policy_{policy_name}{i+1}',counting_range)
        stag_sum+=sum(prey_tuple[0])
        hare_sum+=sum(prey_tuple[1])
        blank_sum+=sum(prey_tuple[2])
    return stag_sum,hare_sum,blank_sum
'''
def get_DataFrame(filename,policy_name,count,class_name,counting_range):
    average_reward_list=get_many_AR_range_of_one_policy(filename,policy_name,count,counting_range)
    list_len=len(average_reward_list)
    episode_len=len(average_reward_list[0])-1
    total_list=[]
    for lst in average_reward_list:
        lst.pop(0)
        total_list.extend(lst)
    data={
    'episode':list(range(counting_range,episode_len+counting_range))*list_len,
    #'episode':list(range(counting_range,counting_range*(episode_len+1),counting_range))*list_len,
    'policy':[f'{class_name}']*(episode_len*list_len),
    'avg_reward':total_list}
    return pd.DataFrame(data)
'''

path='../result/data/original_data/0_after_first_hunt/'
file_list=['progress_MCTS_0steps.csv']
for id in range(len(file_list)):
    file_list[id]=path+file_list[id]
counting_range=300
prey_num=get_many_AR_range_of_one_policy(file_list[0],'ToM',4,counting_range)
print(prey_num)
s=sum(prey_num)
ratio=[0,0,0]
for i in range(len(prey_num)):
    ratio[i]=prey_num[i]/s
print(ratio)

