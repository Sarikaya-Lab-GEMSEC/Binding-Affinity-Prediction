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

epochs = 100

#initialize data and tensors 
print('Initializing Data and Tensors...')
data={}
dirname='C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\'
for i in [1,2,3]:
    data['set'+str(i)]=pd.read_csv(dirname+'All_peptides_Set'+str(1)+'.csv', engine='python')
    data['set'+str(i)].set_index('AA_seq',inplace=True)
    data['set'+str(i)]['Total']=data['set'+str(i)]['CE']+data['set'+str(i)]['CP1']+data['set'+str(i)]['CP2']+data['set'+str(i)]['CP3']
    data['set'+str(i)]=data['set'+str(i)][data['set'+str(i)].Total>=4]
all_seq = pd.concat([data['set1'], data['set2'], data['set3']])
names = all_seq.index.values.tolist()
affinities = all_seq['binding_affinity']


name_train = names[:(len(names) // 10) * 8]
affin_train = affinities[:(len(affinities) // 10) * 8]

name_val = names[(len(names) // 10) * 8:]
affin_val = affinities[(len(affinities ) // 10) * 8:]

    

    
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
sm=Var(torch.randn(20,1),requires_grad=True) #initalize similarity matrix - random array of 20 numbers
sm=torch.mm(sm,sm.t()) #make simalirity matrix square symmetric 
freq_m=Var(torch.randn(12,20),requires_grad=True)
freq_m.data=(freq_m.data-freq_m.min().data)/(freq_m.max().data-freq_m.min().data)#0 to 1 scaling
freq_m.data=freq_m.data/freq_m.data.sum(1,keepdim=True) #sum of each row must be 1 (sum of probabilities of each amino acid at each position)
#loss = nn.MSELoss()   
optimizer = optim.SGD([torch.nn.Parameter(sm), torch.nn.Parameter(freq_m)], lr=1e-4)
#optimizer = optim.SGD([freq_m, sm], lr=1e-4)



#training loop  
loss = nn.MSELoss()
for i in range(epochs): 
    print('Epoch: ' + str(i + 1))
    #forward pass    
    error_list = [1000]
    top_s = None 
    top_fm = None 
    for j, seq in enumerate(name_train):
        affin = affin_train[j]
        new_m = torch.mm(p_one_hot(seq), freq_m)
        tss_m = new_m * sm
        tss_score = tss_m.sum()
        pred = loss(tss_score, torch.FloatTensor(torch.Tensor([affin])))
        error = pow(abs(tss_score - affin), 2)
        if error < min(error_list): 
           top_s = sm 
           top_fm = freq_m 
        error_list.append(error)
        sys.stdout.flush()
        print('On iteration ' + str(j + 1) + ' out of ' + str(len(name_train)) + '. Error: ' + str(error.item()), end='\r')
    print('Lowest Error: ' + str(min(error_list)))
    sm = top_s
    freq_m = top_fm    
    torch.save(sm, 'C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\sm')
    torch.save(freq_m, 'C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\freq_m') 
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

best_sm = sm
best_freq = freq_m
torch.save(best_sm, 'C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\best_sim')
torch.save(best_freq, 'C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\best_frequency') 