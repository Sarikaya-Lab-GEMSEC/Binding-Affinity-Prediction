#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:20:52 2019
@author: FareedMabrouk
"""
from __future__ import print_function
import torch
import pandas as pd
from torch.autograd import Variable as Var
import torch.optim as optim
import torch.nn as nn
import numpy as np 
import sys
import pickle
import statistics 
import scipy.stats

epochs = 100


#initialize data and tensors 
print('Initializing Data and Tensors...')
data={}
windows = 'C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\'
mac = '/Users/FareedMabrouk/Desktop/Explore/Work/GEMSEC/PyTorch/Binding-Affinity-Prediction/'
ubuntu = '/home/gromacs/Desktop/Binding-Affinity-Prediction/'
dirname = mac
for i in [1,2,3]:
    data['set'+str(i)]=pd.read_csv(dirname+'All_peptides_Set'+str(1)+'.csv', engine='python')
    data['set'+str(i)].set_index('AA_seq',inplace=True)
    data['set'+str(i)]['Total']=data['set'+str(i)]['CE']+data['set'+str(i)]['CP1']+data['set'+str(i)]['CP2']+data['set'+str(i)]['CP3']
    data['set'+str(i)]=data['set'+str(i)][data['set'+str(i)].Total>=4]
all_seq = pd.concat([data['set1'], data['set2'], data['set3']]) 

#one hot encoding
AA=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
loc=['N','2','3','4','5','6','7','8','9','10','11','C']
aa = "ARNDCQEGHILKMFPSTWYV"
def p_one_hot(seq):
    c2i = dict((c,i) for i,c in enumerate(aa))
    int_encoded = [c2i[char] for char in seq]
    onehot_encoded = list()
    for value in int_encoded:
        letter = [0 for _ in range(len(aa))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return(torch.Tensor(np.transpose(onehot_encoded)))
   
#initialize tensors 
a=Var(torch.randn(20,1),requires_grad=True) #initalize similarity matrix - random array of 20 numbers
freq_m=Var(torch.randn(12,20),requires_grad=True)
freq_m.data=(freq_m.data-freq_m.min().data)/(freq_m.max().data-freq_m.min().data)#0 to 1 scaling
#loss = nn.MSELoss()   
optimizer = optim.SGD([a, freq_m], lr=1e-4)
#optimizer = optim.SGD([freq_m, sm], lr=1e-4)

trend_line = []
for i in range(10000):
    trend_line.append(-1 * pow(i, 3))

#training loop  
loss = nn.MSELoss()
top_s = None
top_fm = None
avg_loss = []
all_error = []
for i in range(epochs): 
    train = all_seq.sample(frac=.03)
    names = train.index.values.tolist()
    affinities = train['binding_affinity']
    print('Epoch: ' + str(i))
    #forward pass    
    iteration_loss=[]
    for j, seq in enumerate(names):
        sm=torch.mm(a,a.t()) #make simalirity matrix square symmetric
        freq_m.data=freq_m.data/freq_m.data.sum(1,keepdim=True) #sum of each row must be 1 (sum of probabilities of each amino acid at each position)
        affin_score = affinities[j]
        new_m = torch.mm(p_one_hot(seq), freq_m)
        tss_m = new_m * sm
        tss_score = tss_m.sum()
        sms = sm
        fms = freq_m
        error = loss(tss_score, torch.FloatTensor(torch.Tensor([affin_score])))
        curr_error = error.item()
        iteration_loss.append(curr_error)
        all_error.append(curr_error)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    trend_line = []
    for i in range(len(iteration_loss)):
        trend_line.append(-1 * pow(i, 3))    
    sp_correlation = scipy.stats.spearmanr(trend_line, iteration_loss)[0]
    mean = statistics.mean(iteration_loss)
    stdev = statistics.stdev(iteration_loss)       
    torch.save(sm, dirname + 'similarity_matrix_tensor')
    torch.save(freq_m, dirname + 'frequency_matrix_tensor')
    np.save(dirname + 'best_similary_np', sm.detach().numpy())
    np.save(dirname + 'best_frequency_np', freq_m.detach().numpy())
    with open("all_errors", "wb") as fp:
        pickle.dump(all_error, fp)
    print('Average Error: ' + str(mean))
    print('Standard Deviation: ' + str(stdev))
    print('Spearmans Rank Correlation: ' + str(sp_correlation))
print('Training Completed')