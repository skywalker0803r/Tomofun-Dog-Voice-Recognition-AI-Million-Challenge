import numpy as np
import pandas as pd
import torch

def softmax2onehot(probs):
  index = probs.index
  columns = probs.columns
  probs = torch.FloatTensor(probs.values)
  max_idx = torch.argmax(probs, 1, keepdim=True)
  one_hot = torch.FloatTensor(probs.shape)
  one_hot.zero_()
  one_hot.scatter_(1, max_idx, 1)
  return pd.DataFrame(one_hot.detach().numpy(),index=index,columns=columns)


df = pd.read_csv('submit.csv')
print(df.head(5))
one_hot_df = softmax2onehot(df.iloc[:,1:])
df.iloc[:,1:] = one_hot_df
print(df.head(5))
df.to_csv('submit_one_hot.csv',index = False)
print('done')
