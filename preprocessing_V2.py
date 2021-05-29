import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os
from spec_augment_pytorch import spec_augment
import librosa
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import OneHotEncoder


def load_X():
    wav_dir = 'train/train/'
    files = os.listdir(wav_dir)
    X = torch.FloatTensor([])
    for f in tqdm(files[:]):
        try:
            audio, sampling_rate = librosa.load(wav_dir+f)
            mfcc = librosa.feature.mfcc(audio, sr=sampling_rate, n_mfcc=13)
            mfcc = np.expand_dims(mfcc,axis=0)
            delta1_mfcc = librosa.feature.delta(mfcc, order=1)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            features = np.concatenate((mfcc, delta1_mfcc, delta2_mfcc), axis=0)
            shape = features.shape
            features = torch.from_numpy(np.reshape(features, (-1, shape[0], shape[1],shape[2])))
            X = torch.cat([X,features],dim=0)
        except:
            print(f,'file error')
    return X

def look_sample(c,X,Y):
    plt.figure(figsize=(20,10))
    for j,i in enumerate(Y.Remark.unique()):
        sample = Y[Y.Remark==i].sample(1)
        idx = sample.index[0]
        label = sample['Label'][0]
        idx = int(idx.split('_')[1])
        plt.subplot(4,4,j+1)
        plt.imshow(X[idx,c,:,:].log2()) # log2方便觀察
        title = 'name:{} label:{} channel"{}'.format(i,label,c)
        plt.title(title)
    plt.tight_layout()
    plt.savefig('sample wight channel {}'.format(c))
    plt.show()
    

# X
X = load_X()

# Y
Y = pd.read_csv('train\meta_train.csv',index_col='Filename')
Y = Y[Y.index != 'train_01046'] # this file have error
enc = OneHotEncoder().fit(Y[['Label']])
Y_one_hot = enc.transform(Y[['Label']]).toarray()

for c in [0,1,2]:
    look_sample(c,X,Y)

# create pytorch dataloader and save
total_data_len = X.shape[0]
train_percent = 0.8
train_size = int(train_percent*total_data_len)
train_idx = list(np.random.choice([*range(total_data_len)],size=train_size,replace=False))
vaild_idx = list(set([*range(total_data_len)]) - set(train_idx))
trainset = TensorDataset(torch.FloatTensor(X)[train_idx],torch.FloatTensor(Y_one_hot)[train_idx])
vaildset = TensorDataset(torch.FloatTensor(X)[vaild_idx],torch.FloatTensor(Y_one_hot)[vaild_idx])
train_iter = DataLoader(trainset,batch_size=64)
vaild_iter = DataLoader(vaildset,batch_size=64)
torch.save(train_iter, 'train_iter.pt')
torch.save(vaild_iter, 'vaild_iter.pt')
print('preprocessing done')
