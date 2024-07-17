import csv
from itertools import count
from math import ceil
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
        sum_reward_list=[[0] for _ in range(num_of_agents)]
        average_reward_list=[[0] for _ in range(num_of_agents)]

        for i in range(len(reward_this_iter)):
            id=i % num_of_agents
            sum_reward_list[id].append(sum_reward_list[id][-1]+reward_this_iter[i])
            length=len(sum_reward_list[id])-1
            '''
            if length % counting_range == 0:
                    average_reward_list[id].append((sum_reward_list[id][-1]-sum_reward_list[id][-1-counting_range])/counting_range)
            '''
            
            if length>=int(counting_range) and length % 3 == 0:
                average_reward_list[id].append((sum_reward_list[id][-1]-sum_reward_list[id][-1-counting_range])/counting_range)
            
            
        for row in reader:
            episodes_this_iter=eval(row['episodes_this_iter'])
            reward_this_iter=eval(row[f'hist_stats/{reward_name}_reward'])
            reward_this_iter=reward_this_iter[-episodes_this_iter*num_of_agents:]
            for i in range(len(reward_this_iter)):
                id=i % num_of_agents
                sum_reward_list[id].append(sum_reward_list[id][-1]+reward_this_iter[i])
                length=len(sum_reward_list[id])-1
                
                if length>2001:
                    return average_reward_list
                
                '''
                if length % counting_range == 0:
                    average_reward_list[id].append((sum_reward_list[id][-1]-sum_reward_list[id][-1-counting_range])/counting_range)
                '''
                
                if length>=int(counting_range) and length % 3 == 0:
                    average_reward_list[id].append((sum_reward_list[id][-1]-sum_reward_list[id][-1-counting_range])/counting_range)
                
    return average_reward_list

def get_many_AR_range_of_one_policy(filename,policy_name,count,counting_range):
    average_reward_list=[]
    for i in range(count):
        average_reward_list.extend(get_AR_range(filename,f'policy_{policy_name}{i+1}',counting_range))
    return average_reward_list

def get_DataFrame(filename,policy_name,count,class_name,counting_range,cb):
    average_reward_list=get_many_AR_range_of_one_policy(filename,policy_name,count,counting_range)
    list_len=len(average_reward_list)
    episode_len=len(average_reward_list[0])-1
    total_list=[]
    for lst in average_reward_list:
        lst.pop(0)
        total_list.extend(lst)
    data={
    'episode':list(range(ceil(counting_range/3)*3,3*episode_len+ceil(counting_range/3)*3,3))*list_len,
    #'episode':list(range(counting_range,counting_range*(episode_len+1),counting_range))*list_len,
    'policy':[f'{class_name}']*(episode_len*list_len),
    'avg_reward':total_list,
    'choice_temperature':[cb]*(episode_len*list_len),}
    return pd.DataFrame(data)


path='../result/data/original_data/5_after_first_hunt/'
record_csv='../result/data/5_after_first_hunt/4ToM_avg200_reward.csv'
file_list=['progress_4MCTS_cb1.csv','progress_4MCTS_cb3.csv','progress_4MCTS_dist_cb1.csv',
'progress_4MCTS_dist_cb3.csv','progress_4multiMCTS_cb1.csv','progress_4multiMCTS_cb3.csv','progress_4MCTS_5trees_cb1.csv']
class_name_list=['ToM_MCTS','ToM_MCTS','ToM_MCTS_dist','ToM_MCTS_dist','ToM_multiMCTS','ToM_multiMCTS','ToM_MCTS_5trees']
policy_number=[4]*len(file_list)
choice_temperature=[1,3,1,3,1,3,1]
policy_list=['ToM']*len(file_list)
for id in range(len(file_list)):
    file_list[id]=path+file_list[id]
counting_range=200

'''
path='../result/data/original_data/5_after_first_hunt/adapt/'
record_csv='../result/data/5_after_first_hunt/adapt/all_avg300_reward_2s1h.csv'
file_list=['progress_1MCTS+2s1h_cb1.csv','progress_1MCTS+2s1h_cb3.csv','progress_1MCTS_dist+2s1h_cb1.csv',
'progress_1MCTS_dist+2s1h_cb3.csv','progress_1multiMCTS+2s1h_cb1.csv','progress_1multiMCTS+2s1h_cb3.csv',
'progress_1MCTS_5trees+2s1h_cb1.csv','progress_1IRL+2s1h.csv']
class_name_list=['ToM_MCTS','ToM_MCTS','ToM_MCTS_dist','ToM_MCTS_dist','ToM_multiMCTS','ToM_multiMCTS','ToM_MCTS_5trees','IRL']
policy_number=[1]*len(file_list)
choice_temperature=[1,3,1,3,1,3,1,'undefined']
policy_list=['ToM']*(len(file_list)-1)+['train']
for id in range(len(file_list)):
    file_list[id]=path+file_list[id]
counting_range=200
'''
'''
concat_list=[get_DataFrame(file_list[0],'ToM',1,'multiMCTS',counting_range),
get_DataFrame(file_list[1],'ToM',1,'MCTS',counting_range),
get_DataFrame(file_list[2],'train',1,'IRL',counting_range),
#get_DataFrame(file_list[3],'ToM',4,'ToM_multiMCTS_type_based',counting_range)
]
'''
concat_list=[get_DataFrame(file_list[i],policy_list[i],policy_number[i],class_name_list[i],
counting_range,choice_temperature[i]) for i in range(len(file_list))]

frame=pd.concat(concat_list,axis=0)
frame.index=list(range(frame.shape[0]))

frame.to_csv(record_csv)


frame=pd.read_csv(record_csv)
#sns.set_context(rc={"lines.linewidth": 1})
sns.lineplot(x='episode',y='avg_reward',hue='policy',style='choice_temperature',data=frame)


plt.xlabel('episodes')
plt.ylabel(f'average reward in last {counting_range} episodes')

plt.title('4p2s_5stepsAfterFirstHunt')

plt.show()