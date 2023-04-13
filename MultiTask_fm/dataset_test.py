from torch.utils.data import DataLoader
from dataset import XDataset

datafile = '/Users/chikuo/Desktop/博士学习/code/推荐系统/Multitask-Recommendation-Library-main/data/AliCCP/ctr_cvr2w.dev'
dataset = XDataset(datafile)

dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
for i, value in enumerate(dataloader):
  click, conversion, deep_conversion, features = value
  print(click.shape)
  print(conversion.shape)
  for key in features.keys():
    print(key, features[key].shape)
