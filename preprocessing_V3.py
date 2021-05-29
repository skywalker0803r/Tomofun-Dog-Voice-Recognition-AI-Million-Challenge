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
    for f in tqdm(files[:5]):
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

# trainset and validset
X_train = load_wav_file_as_model_input('train/train/')
Y_train = pd.read_csv('train/meta_train.csv',index_col='Filename')
Y_train = Y_train[Y_train.index != 'train_01046'] # this file have error
enc = OneHotEncoder().fit(Y_train[['Label']])
Y_train_one_hot = enc.transform(Y_train[['Label']]).toarray()
total_data_len = X_train.shape[0]
train_percent = 0.8
train_size = int(train_percent*total_data_len)
train_idx = list(np.random.choice([*range(total_data_len)],size=train_size,replace=False))
vaild_idx = list(set([*range(total_data_len)]) - set(train_idx))
train_iter = DataLoader(TensorDataset(torch.FloatTensor(X_train)[train_idx],torch.FloatTensor(Y_train_one_hot)[train_idx]),batch_size=64)
vaild_iter = DataLoader(TensorDataset(torch.FloatTensor(X_train)[vaild_idx],torch.FloatTensor(Y_train_one_hot)[vaild_idx]),batch_size=64)
torch.save(train_iter, 'train_iter.pt')
torch.save(vaild_iter, 'vaild_iter.pt')

# testset
X_test = load_wav_file_as_model_input('public_test/public_test/')
test_iter = DataLoader(TensorDataset(torch.FloatTensor(X_test)),batch_size=1)
torch.save(test_iter, 'test_iter.pt')

print('preprocessing done')