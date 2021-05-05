# -*- coding: utf-8 -*-
"""
@author: nkzc8888
"""

import numpy as np
from torch import nn, optim
import torch
import data_processing
from torch.utils.data import DataLoader
import cnn
import visdom
import pandas as pd
vis=visdom.Visdom()

data = data_processing.load_data(download=False)
new_data = data_processing.convert2onehot(data)

new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data)
sep = int(0.7*len(new_data))
train_data = new_data[:sep]
test_data = new_data[sep:]

batch_size = 64
learning_rate = 0.1
num_epochs = 300

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

model = cnn.CNN()
#model = model.cuda()

#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

epoch = 0
plot_data = {'X': [], 'Y': []}
for epoch in range(num_epochs):
    for data in train_loader:
        #data = train_data
        img, label = data[:,:21], data[:,21:]
        img, label = torch.tensor(img), torch.tensor(label)
        #img, label = img.cuda(), label.cuda()
        
    
        N = label.size(0)
    
        out = model(img)
    
        log_prob = torch.nn.functional.log_softmax(out, dim=1)
        loss = -torch.sum(log_prob * label) / N
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    plot_data['Y'].append(print_loss)
    plot_data['X'].append(epoch)
    
    if epoch%50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
                
    if epoch == 200:
        vis.line(X=np.array(plot_data['X']),Y=np.array(plot_data['Y']),opts={
                    'title':  ' loss over time',
                    'legend': ['loss'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'})



num_correct = 0
for data in test_loader:

    img, label = data[:,:21], data[:,21:]
    img, label = torch.tensor(img), torch.tensor(label)
    #img, label = img.cuda(), label.cuda()
    
    N = label.size(0)
    out = model(img)
    log_prob = torch.nn.functional.log_softmax(out, dim=1)
    _,pred = torch.max(out, 1)
    
    pred = pd.get_dummies(pred)
    pred = pred.values.astype(np.float32)
    pred = torch.tensor(pred)

    for i in range(len(pred)):
        if (pred[i] == label[i]).all():
            num_correct += 1
    
print('Test Acc: {:.6f}'.format(
    num_correct / (len(test_data))
))
