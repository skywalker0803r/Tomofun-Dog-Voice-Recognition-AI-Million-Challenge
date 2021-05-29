import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os
import librosa
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import OneHotEncoder
import random

def sample2melspectrogram(samples,sample_rate):
    melspectrogram = librosa.feature.melspectrogram(samples,sample_rate,center=False)
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    melspectrogram = (melspectrogram - melspectrogram.min()) / (melspectrogram.max() - melspectrogram.min())
    melspectrogram = melspectrogram[:80,:]
    return melspectrogram

def load_wav_file_as_model_input(wav_dir):
    files = os.listdir(wav_dir)
    X = torch.FloatTensor([])
    for f in tqdm(files[:]):
        try:
            samples, sample_rate = librosa.load(wav_dir+f)
            mel_spectrogram = sample2melspectrogram(samples,sample_rate)
            shape = mel_spectrogram.shape
            mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))
            mel_spectrogram = torch.from_numpy(mel_spectrogram)
            X = torch.cat([X,torch.unsqueeze(mel_spectrogram,0)],dim=0)
        except:
            print('the file:{} have problem'.format(f))
    # X shape = batch,channel,length,width
    return X 

def load_wav_file_from_public_test(files):
    X = torch.FloatTensor([])
    for f in tqdm(files[:]):
        samples, sample_rate = librosa.load('public_test/public_test/'+f)
        mel_spectrogram = sample2melspectrogram(samples,sample_rate)
        shape = mel_spectrogram.shape
        mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))
        mel_spectrogram = torch.from_numpy(mel_spectrogram)
        X = torch.cat([X,torch.unsqueeze(mel_spectrogram,0)],dim=0)
    return X
        
# load trainset
X_train = load_wav_file_as_model_input('train/train/') # load all data
Y_train = pd.read_csv('train/meta_train.csv',index_col='Filename')
Y_train = Y_train[Y_train.index != 'train_01046'] # this file have error
enc = OneHotEncoder().fit(Y_train[['Label']]) # one-hot encoding
Y_train_one_hot = enc.transform(Y_train[['Label']]).toarray()

# testset not only inference but also can use training model
Y_test = pd.read_csv('public_test/meta_test.csv',index_col='Filename').dropna(axis=0) # only load have label data
print(Y_test)
Y_test_one_hot = enc.transform(Y_test[['Label']]).toarray() # one_hot encoding
X_test = load_wav_file_from_public_test([f+'.wav' for f in Y_test.index])# only load have label data

# convert to tensor and merge togather
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
X = torch.cat([X_train,X_test],dim=0)
print('X.shape',X.shape)
# convert to tensor and merge togather
Y_train = torch.FloatTensor(Y_train_one_hot)
Y_test = torch.FloatTensor(Y_test_one_hot)
Y = torch.cat([Y_train,Y_test],dim=0)
print('Y.shape',Y.shape)
# train vaild split
total_data_len = X.shape[0]
train_percent = 0.8
train_size = int(train_percent*total_data_len)
train_idx = list(np.random.choice([*range(total_data_len)],size=train_size,replace=False))
vaild_idx = list(set([*range(total_data_len)]) - set(train_idx))
trainset = TensorDataset(X[train_idx],Y[train_idx])
vaildset = TensorDataset(X[vaild_idx],Y[vaild_idx])
train_iter = DataLoader(trainset,batch_size=64)
vaild_iter = DataLoader(vaildset,batch_size=64)

# print infomation
print('number of train:',len(trainset))
print('number of vaild:',len(vaildset))

# save dataloader for modeling
torch.save(train_iter, 'train_iter.pt')
torch.save(vaild_iter, 'vaild_iter.pt')
print('preprocessing done')
