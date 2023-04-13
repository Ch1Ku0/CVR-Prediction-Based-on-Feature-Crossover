import torch

import pandas as pd


# a = torch.ones([8, 4, 5, 6])
# print('a =',a.size())
# b = torch.ones([5, 6])
# print('b =',b.size())
# c = a+b
# print('c =',c.size())



# b= torch.tensor([[ 3.8332e-05],
#         [-1.9330e-04]])
# c= torch.tensor([[[0.4951, 0.4996, 0.5007, 0.4987],
#          [0.4980, 0.5009, 0.4999, 0.5013],
#          [0.5013, 0.4996, 0.5027, 0.5014],
#          [0.5000, 0.5033, 0.5014, 0.4998],
#          [0.5003, 0.5016, 0.5010, 0.5012]],
#
#         [[0.5004, 0.4972, 0.4973, 0.4971],
#          [0.5035, 0.4973, 0.4999, 0.4969],
#          [0.4973, 0.5006, 0.4989, 0.4948],
#          [0.5022, 0.5049, 0.4985, 0.4977],
#          [0.5035, 0.5025, 0.4973, 0.4978]]])
#
# print(b.size())
# print(c.size())
#
# b = torch.unsqueeze(b, -1)
# #
# b = b.repeat(1,c.size()[1],c.size()[2])
# #
# print(b.size())
#
# d = b + c
# print(d)


# t = torch.tensor([[[1, 2, 2, 1],
#                    [3, 4, 4, 3],
#                    [1, 2, 3, 4]],
#                   [[5, 6, 6, 5],
#                    [7, 8, 8, 7],
#                    [5, 6, 7, 8]]])
# print(t, t.shape)
#
# x = torch.flatten(t, start_dim=1)
# print(x, x.shape)
#
# vocabulary_size = {'101': 238635, '121': 98, '122': 14, '124': 3, '125': 8, '126': 4, '127': 4, '128': 3, '129': 5,
#                    '205': 467298, '206': 6929, '207': 263942, '216': 106399, '508': 5888, '509': 104830, '702': 51878,
#                    '853': 37148, '301': 4}


train_df = pd.read_csv(
        '/Users/chikuo/Desktop/博士学习/code/推荐系统/Multitask-Recommendation-Library-main/data/AliCCP/ctr_cvr2w.train',
        delimiter=',',
        header=None,
        index_col=None,
        # names=column_names
    )

# print(train_df)
for idx, col in enumerate(train_df.columns):
    print(str(col) + '——————' + str(len(train_df[col].unique()) + 1))