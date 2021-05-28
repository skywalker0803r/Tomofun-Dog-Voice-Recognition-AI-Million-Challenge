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
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset,DataLoader
import numpy as np

def load_X(augment=False):
    wav_dir = 'train/train/'
    files = os.listdir(wav_dir)
    X = torch.FloatTensor([])
    for f in tqdm(files[:]):
        try:
            # load audio
            audio, sampling_rate = librosa.load(wav_dir+f)
            # mel_spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sampling_rate,n_mels=256,hop_length=128,fmax=8000)
            # reshape spectrogram shape to [batch_size, time, frequency]
            shape = mel_spectrogram.shape
            mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))
            mel_spectrogram = torch.from_numpy(mel_spectrogram)
            # data Augment if you want
            if augment == True:
                # augment method reference:
                # https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html?m=1&fbclid=IwAR27ucBjiXdRoRwVAmQfVBK7-4mN4Ln_lAFL11-ChqbnLUiDxDsNM4wesEA
                # https://github.com/DemisEom/SpecAugment
                mel_spectrogram = spec_augment(mel_spectrogram=mel_spectrogram)
            # append to X
            X = torch.cat([X,torch.unsqueeze(mel_spectrogram,0)],dim=0)
        except:
            print('the file:{} have problem'.format(f))
            pass   
    return X

# X 
X_without_specaugment = load_X(augment = False) # 一份原本的
X_specaugment = [load_X(augment = True) for i in range(10)] #10份specaugment後的
# 原本的specaugment後的X作合併 1+10 = 11份
X = X_without_specaugment
for i in range(10):
    X = torch.cat([X,X_specaugment[i]],dim=0)

# Y with one_hot
Y = pd.read_csv('train\meta_train.csv',index_col='Filename')
Y = Y[Y.index != 'train_01046']
enc = OneHotEncoder()
enc.fit(Y[['Label']])
Y_one_hot = enc.transform(Y[['Label']]).toarray()
# 複製11份
Y = np.array([])
for i in range(11):
  if len(Y) == 0:
    Y = Y_one_hot
  else:
    Y = np.vstack((Y,Y_one_hot))

# create pytorch dataloader and save
total_data_len = X.shape[0]
train_percent = 0.8
train_size = int(train_percent*total_data_len)
train_idx = list(np.random.choice([*range(total_data_len)],size=train_size,replace=False))
vaild_idx = list(set([*range(total_data_len)]) - set(train_idx))
trainset = TensorDataset(torch.FloatTensor(X)[train_idx],torch.FloatTensor(Y)[train_idx])
vaildset = TensorDataset(torch.FloatTensor(X)[vaild_idx],torch.FloatTensor(Y)[vaild_idx])
train_iter = DataLoader(trainset,batch_size=64)
vaild_iter = DataLoader(vaildset,batch_size=64)
torch.save(train_iter, 'train_iter.pt')
torch.save(vaild_iter, 'vaild_iter.pt')
print('preprocessing done')