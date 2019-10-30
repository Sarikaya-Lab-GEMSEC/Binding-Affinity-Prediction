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

epochs = 100

#initialize data and tensors 
print('Initializing Data and Tensors...')
data={}
#windows dirname: 'C:\\Users\\GEMSEC-User\\Desktop\\Fareed_Training_Loop\\'
#mac dirname: /Users/FareedMabrouk/Desktop/Explore/Work/GEMSEC/PyTorch/Binding-Affinity-Prediction/
#ubuntu dirname: /home/gromacs/Desktop/Fareed_Training_Loop
dirname='/home/gromacs/Desktop/Fareed_Training_Loop/'
for i in [1,2,3]:
    data['set'+str(i)]=pd.read_csv(dirname+'All_peptides_Set'+str(1)+'.csv', engine='python')
    data['set'+str(i)].set_index('AA_seq',inplace=True)
    data['set'+str(i)]['Total']=data['set'+str(i)]['CE']+data['set'+str(i)]['CP1']+data['set'+str(i)]['CP2']+data['set'+str(i)]['CP3']
    data['set'+str(i)]=data['set'+str(i)][data['set'+str(i)].Total>=4]
all_seq = pd.concat([data['set1'], data['set2'], data['set3']])
train=all_seq.sample(frac=0.1) 
names = train.index.values.tolist()
affinities = train['binding_affinity']
test=all_seq.drop(train.index)    

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

#try training with 10 percent, figure how long it's going to take, if it takes about 3 days, test that then run it on 90 percent of the set 

#training loop  
loss = nn.MSELoss()
top_s = None
top_fm = None
avg_loss = []
epoch_loss=[]
minimum_error = 1000;
for i in range(epochs): 
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
        #^ needs to fit affinity trend 
        sms = sm
        fms = freq_m
        error = loss(tss_score, torch.FloatTensor(torch.Tensor([affin_score])))
        iteration_loss.append(error.item())
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        i_length = len(iteration_loss)
        i_sum = sum(iteration_loss)
        if i_length == (20000):
            avg_loss.append(i_sum / i_length)
            iteration_loss.clear()
        if i_length > 10000 and error.item() < min(iteration_loss, default=999): 
           top_s = np.asarray(sms.detach())
           top_fm = np.asarray(fms.detach())
        sys.stdout.flush()
    torch.save(sm, dirname + 'similarity_to_test_10percent')
    torch.save(freq_m, dirname + 'frequency_to_test10percent')
    with open("iteration_loss", "wb") as fp:   #Pickling
        pickle.dump(iteration_loss, fp)
    print('Last Error: ' + str(iteration_loss[-1]))
np.save(dirname + 'bestsm1', top_s)
np.save(dirname + 'best_freq1', top_fm)