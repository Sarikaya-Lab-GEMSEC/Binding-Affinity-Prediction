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
a=Var(torch.randn(20,1),requires_grad=True) #initalize similarity matrix - random array of 20 numbers
freq_m=Var(torch.randn(12,20),requires_grad=True)
freq_m.data=(freq_m.data-freq_m.min().data)/(freq_m.max().data-freq_m.min().data)#0 to 1 scaling
#loss = nn.MSELoss()   
optimizer = optim.SGD([torch.nn.Parameter(a), torch.nn.Parameter(freq_m)], lr=1e-4)
#optimizer = optim.SGD([freq_m, sm], lr=1e-4)

#flush error list every 100? 


#training loop  
loss = nn.MSELoss()
top_s = None
top_fm = None
epoch_loss=[]
for i in range(epochs): 
    print('Epoch: ' + str(i))
    #forward pass    
    iteration_loss=[]
    for j, seq in enumerate(name_train):
        sm=torch.mm(a,a.t()) #make simalirity matrix square symmetric
        freq_m.data=freq_m.data/freq_m.data.sum(1,keepdim=True) #sum of each row must be 1 (sum of probabilities of each amino acid at each position)
        affin_score = affin_train[j]
        new_m = torch.mm(p_one_hot(seq), freq_m)
        tss_m = new_m * sm
        tss_score = tss_m.sum()
        sms = sm
        fms = freq_m
        error = loss(tss_score, torch.FloatTensor(torch.Tensor([affin_score])))
        if len(iteration_loss) == 1000:
            iteration_loss = [1000]
        if len(iteration_loss) > 500 and error.item() < min(iteration_loss): 
           top_s = np.asarray(sms.detach())
           top_fm = np.asarray(fms.detach())
        iteration_loss.append(error.item())
        sys.stdout.flush()
        print('Epoch: '+str(i)+' On iteration ' + str(j + 1) + ' out of ' + str(len(name_train)) + '. Error: ' + str(error.item()), end='\r')

        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        np.save('C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\sm', top_s)
        np.save('C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\freq_m', top_fm)
    epoch_loss.append(np.mean(iteration_loss))
    
torch.save(top_s, 'C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\best_sim')
torch.save(top_fm, 'C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\best_frequency') 