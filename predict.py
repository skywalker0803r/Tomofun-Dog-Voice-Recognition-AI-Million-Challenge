import torch
from torch import nn
import pandas as pd
import os
from tqdm import tqdm
import torchaudio

model = torch.load('model.pt')

test_data_dir = 'public_test/public_test/'

files = os.listdir(test_data_dir)
X = torch.FloatTensor([])
for f in tqdm(files[:]):
    # 聲音訊號
    waveform, sample_rate = torchaudio.load(test_data_dir+f)
    # 聲音頻譜圖
    specgram = torchaudio.transforms.Spectrogram()(waveform)
    specgram = torch.unsqueeze(specgram, 0)
    # 加入X
    X = torch.cat([X,specgram],dim=0)

# 推論
X = torch.FloatTensor(X)
y_hat = model(X).detach().numpy()
print(y_hat.shape)

submit = pd.DataFrame(columns=['Filename','Barking','Howling','Crying','COSmoke','GlassBreaking','Other'])
submit['Filename'] = [ f.split('.')[0] for f in files[:]]
submit.iloc[:,1:] = y_hat
submit.to_csv('submit.csv')
print('done')