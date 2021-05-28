import torch
from torch import nn
import pandas as pd
import os
from tqdm import tqdm
import torchaudio
import librosa

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

model = torch.load('model.pt').to(device)

test_data_dir = 'public_test/public_test/'

files = os.listdir(test_data_dir)
X = torch.FloatTensor([])
for f in tqdm(files[:]):
    # load audio
    audio, sampling_rate = librosa.load(test_data_dir+f)
     # mel_spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sampling_rate,n_mels=256,hop_length=128,fmax=8000)
    # reshape spectrogram shape to [batch_size, time, frequency]
    shape = mel_spectrogram.shape
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))
    mel_spectrogram = torch.from_numpy(mel_spectrogram)
    X = torch.cat([X,torch.unsqueeze(mel_spectrogram,0)],dim=0)

# inference and output submit.csv
X = torch.FloatTensor(X).to(device)
y_hat = model(X).detach().cpu().numpy()
sample_submit = pd.read_csv('sample_submission.csv')
sample_submit.iloc[:10000,:] = y_hat
sample_submit.to_csv('submit.csv',index=False)
print('done')