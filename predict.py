import torch
from torch import nn
import pandas as pd
import os
from tqdm import tqdm
import torchaudio
import librosa
import numpy as np
import gc

def sample2melspectrogram(samples,sample_rate):
    melspectrogram = librosa.feature.melspectrogram(samples,sample_rate,center=False)
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    melspectrogram = (melspectrogram - melspectrogram.min()) / (melspectrogram.max() - melspectrogram.min())
    melspectrogram = melspectrogram[:80,:]
    return melspectrogram

# load model
model = torch.load('model.pt').cuda()
print('use model is:',model)

# test_data_dir
test_data_dir = 'public_test/public_test/'

# inference for loop
files = os.listdir(test_data_dir)
n = 10000
sample_submit = pd.read_csv('sample_submission.csv')
i = 0
for f in tqdm(files[:n]):
    # load audio
    samples,sample_rate = librosa.load(test_data_dir+f)
    mel_spectrogram = sample2melspectrogram(samples,sample_rate)
    shape = mel_spectrogram.shape
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))
    X = torch.from_numpy(mel_spectrogram)
    X = torch.unsqueeze(X,0).cuda()
    y_hat = model(X).detach().cpu().numpy()
    sample_submit.iloc[[i],1:] = y_hat
    i += 1
    gc.collect()

# save
sample_submit.to_csv('submit.csv',index=False)
print('done')
