import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from math import ceil

MODE=2
END_EPISODE=8500000
INTERVAL=10000

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
            if MODE==1:
                if length % counting_range == 0:
                        average_reward_list[id].append((sum_reward_list[id][-1]-sum_reward_list[id][-1-counting_range])/counting_range)
            elif MODE==2:
                if length>int(counting_range) and length % INTERVAL==0:
                    average_reward_list[id].append((sum_reward_list[id][-1]-sum_reward_list[id][-1-counting_range])/counting_range)
            
            
        for row in reader:
            episodes_this_iter=eval(row['episodes_this_iter'])
            reward_this_iter=eval(row[f'hist_stats/{reward_name}_reward'])
            reward_this_iter=reward_this_iter[-episodes_this_iter*num_of_agents:]
            for i in range(len(reward_this_iter)):
                id=i % num_of_agents
                sum_reward_list[id].append(sum_reward_list[id][-1]+reward_this_iter[i])
                length=len(sum_reward_list[id])-1
                
                if length>END_EPISODE:
                    return average_reward_list
                if MODE==1:
                    if length % counting_range == 0:
                        average_reward_list[id].append((sum_reward_list[id][-1]-sum_reward_list[id][-1-counting_range])/counting_range)
                elif MODE==2:
                    if length>int(counting_range) and length % INTERVAL==0:
                        average_reward_list[id].append((sum_reward_list[id][-1]-sum_reward_list[id][-1-counting_range])/counting_range)
                
    return average_reward_list

def get_many_AR_range_of_one_policy(filename,policy_name,count,counting_range):
    average_reward_list=[]
    for i in range(count):
        average_reward_list.extend(get_AR_range(filename,f'policy_{policy_name}{i+1}',counting_range))
    return average_reward_list

def get_DataFrame(filename,policy_name,count,class_name,counting_range):
    average_reward_list=get_many_AR_range_of_one_policy(filename,policy_name,count,counting_range)
    list_len=len(average_reward_list)
    episode_len=len(average_reward_list[0])-1
    total_list=[]
    for lst in average_reward_list:
        lst.pop(0)
        total_list.extend(lst)
    if MODE==1:
        data={
        'episode':list(range(counting_range,counting_range*(episode_len+1),counting_range))*list_len,
        'policy':[f'{class_name}']*(episode_len*list_len),
        'avg_reward':total_list}
    elif MODE==2:
        data={
        'episode':list(range(ceil(counting_range/INTERVAL)*INTERVAL,(ceil(counting_range/INTERVAL)+episode_len)*INTERVAL, INTERVAL))*list_len,
        'policy':[f'{class_name}']*(episode_len*list_len),
        'avg_reward':total_list}
    return pd.DataFrame(data)

def get_SA_DataFrame(filename,policy_name,class_name,counting_range):
    average_reward_list=get_AR_range(filename,policy_name,counting_range)[0]
    list_len=1
    episode_len=len(average_reward_list)-1
    average_reward_list.pop(0)
    if MODE==1:
        data={
        'episode':list(range(counting_range,counting_range*(episode_len+1),counting_range))*list_len,
        'policy':[f'{class_name}']*(episode_len*list_len),
        'avg_reward':average_reward_list}
    elif MODE==2:
        data={
        'episode':list(range(counting_range,episode_len+counting_range))*list_len,
        'policy':[f'{class_name}']*(episode_len*list_len),
        'avg_reward':average_reward_list}
    return pd.DataFrame(data)

path='../result/data/original_data/AlphaZeroTrainer/'
record_csv='../result/data/AlphaZeroTrainer/RL_LSTM+4p+5fh.csv'
file_list=['RL_LSTM+4p+5fh.csv']
counting_range=10000

for id in range(len(file_list)):
    file_list[id]=path+file_list[id]

concat_list=[get_DataFrame(file_list[0],'train',4,'LSTM',counting_range),
#get_DataFrame(file_list[1],'train',1,'RL',counting_range),
#get_DataFrame(file_list[2],'ToM',2,'temperature=6',counting_range),
#get_SA_DataFrame(file_list[3],'ToM','ToM_multiMCTS_type_based',counting_range)
]
frame=pd.concat(concat_list,axis=0)
frame.index=list(range(frame.shape[0]))

frame.to_csv(record_csv)

frame=pd.read_csv(record_csv)
sns.lineplot(x='episode',y='avg_reward',hue='policy',data=frame)


plt.xlabel('episodes')
plt.ylabel(f'average reward in last {counting_range} episodes')
plt.title('RL_LSTM+4p+5fh')

plt.show()
