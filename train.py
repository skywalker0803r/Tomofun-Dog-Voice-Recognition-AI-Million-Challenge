import torch
from model import Model
from torch.optim import Adam
from torch import nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

# load data_iter
train_iter = torch.load('train_iter.pt')
vaild_iter = torch.load('vaild_iter.pt')


# train_step
def train_step(model,train_iter,loss_fn,optimizer):
        model.train()
        total_loss = 0
        for t,(x,y) in enumerate(train_iter):
            y_hat = model(x)
            loss = loss_fn(y_hat,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss/(t+1)

# valid_step  
def valid_step(model,vaild_iter,loss_fn,optimizer):
    model.eval()
    total_loss = 0
    for t,(x,y) in enumerate(vaild_iter):
        y_hat = model(x)
        loss = loss_fn(y_hat,y)
        total_loss += loss.item()
    return total_loss/(t+1)


# train_loop
def train(model,optimizer,loss_fn,max_epochs=300,log_interval=10):
        history = {'train_loss':[],'valid_loss':[]}
        current_loss = np.inf
        best_model = None
        for i in tqdm(range(max_epochs)):
            history['train_loss'].append(train_step(model,train_iter,loss_fn,optimizer))
            history['valid_loss'].append(valid_step(model,vaild_iter,loss_fn,optimizer))
            
            if i % log_interval == 0:
                print("epoch:{} train_loss:{:.4f} valid_loss:{:.4f}".format(
                    i,history['train_loss'][-1],history['valid_loss'][-1]))
            
            if history['valid_loss'][-1] <= current_loss:
                best_model = deepcopy(model.eval())
                current_loss = history['valid_loss'][-1]
        model = deepcopy(best_model.eval())
        plt.plot(history['train_loss'],label='train_loss')
        plt.plot(history['valid_loss'],label='valid_loss')
        plt.legend()
        plt.show()
        return model








if __name__ == '__main__':
    model = Model()
    optimizer = Adam(model.parameters(),lr=1e-3)
    loss_fn = nn.MSELoss()
    model = train(model,optimizer,loss_fn,max_epochs=100,log_interval=10)
    torch.save(model,'model.pt')