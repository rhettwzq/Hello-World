# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:24:23 2018

@author: WZQ
"""
from numpy import empty
from hmm import HMM
observations = ('dry','dryish','damp','soggy')

start_probability ={'sunny':0.6,'cloudy':0.2,'rainy':0.2}

transition_probability = {
    'sunny':{'sunny':0.5,'cloudy':0.375,'rainy':0.125},
    'cloudy':{'sunny':0.25,'cloudy':0.125,'rainy':0.625},
    'rainy':{'sunny':0.25,'cloudy':0.375,'rainy':0.375}
}

emission_probability ={
    'sunny':{'dry':0.6,'dryish':0.2,'damp':0.15,'soggy':0.05},
    'cloudy':{'dry':0.25,'dryish':0.25,'damp':0.25,'soggy':0.25},
    'rainy':{'dry':0.05,'dryish':0.1,'damp':0.35,'soggy':0.50}
}
def gengerate_index_map(labels):
    """为状态创建索引编号"""
    index_label = {}
    label_index = {}

    i =0
    for l in labels:
        index_label[i] =l
        label_index[l] =i

        i+=1
    return label_index,index_label
states=['sunny','cloudy','rainy']
# 生成状态索引
states_label_index,states_index_label = gengerate_index_map(states)
# 生成输出序列索引
observations_label_index,observations_index_label = gengerate_index_map(observations)

def convert_observations_to_index(observations,label_index):
    """将观测序列转换成索引编号"""
    list =[]
    for o in observations:
        list.append(label_index[o])
    return list


def convert_map_to_matrix(map,label_index1,label_index2):
    """将状态转移嵌套字典转换为矩阵"""
    m = empty((len(label_index1),len(label_index2)),dtype = float)
    for line in map:
        for col in map[line]:
            m[label_index1[line]][label_index2[col]] = map[line][col]
    return m

def convert_map_to_vector(map,label_index):
    """初始状态转化为数字形式"""
    v = empty(len(map),dtype = float)
    for e in map:
        v[label_index[e]] =map[e]
    return v

A = convert_map_to_matrix(transition_probability,states_label_index,states_label_index)
print (A)
B = convert_map_to_matrix(emission_probability,states_label_index,observations_label_index)
print (B)
observations_index = convert_observations_to_index(observations,observations_label_index)
print (observations_index)
pi = convert_map_to_vector(start_probability,states_label_index)
print (pi)

h = HMM(A,B,pi)
# 人为定义的海藻状态序列
obs_seq = ('dry','damp','soggy')
obs_seq_index = convert_observations_to_index(obs_seq,observations_label_index)

# 计算P(o|lambda)
F = h._forward(obs_seq_index)
print ("forward: P(O|lambda) = %f" %sum(F[:,-1]))
X = h._backward(obs_seq_index)
print ("backward: P(O|lambda) = %f" %sum(X[:,0]*pi*B[:,0]))
