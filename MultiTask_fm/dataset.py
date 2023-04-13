from torch.utils.data import DataLoader
from torch.utils.data import Dataset

label_num = 3
full_fea_num = 60000
class XDataset(Dataset):
  '''load csv data with feature name ad first row'''

  def __init__(self, datafile):
    super(XDataset, self).__init__()
    self.feature_names = []
    self.datafile = datafile
    self.data = []
    self.fea_data = []
    self.data_label = []
    self._load_data()

  def _load_data(self):
    print("start load data from: {}".format(self.datafile))
    self.feature_names = [str(i + 1) for i in range(full_fea_num)]
    with open(self.datafile) as f:
      for line in f:
        res_line = [0 for i in range(full_fea_num)]
        temp_line_list = line.replace('\n', '').split(' ')
        # 这第一个是标签
        label_click = temp_line_list[0]
        label_conversion = temp_line_list[1]
        label_deep_conversion = temp_line_list[2]
        res_label_line = [0 for i in range(label_num)]
        res_label_line[0] = label_click
        res_label_line[1] = label_conversion
        res_label_line[2] = label_deep_conversion
        self.data_label.append(res_label_line)
        # 除了第一个标签，后面都是特征
        temp_line_list_without_label = temp_line_list[3:]

        for fea in temp_line_list_without_label:
          single_fea = fea.split(':')
          res_line[int(single_fea[0])] = int(single_fea[1])
        self.data.append(res_line)
    print("load data from {} finished".format(self.datafile))

  def __len__(self, ):
    return len(self.data)

  def __getitem__(self, idx):
    line = self.data[idx]
    label_line = self.data_label[idx]

    click = label_line[0]
    conversion = label_line[1]
    deep_conversion = label_line[2]

    features = dict(zip(self.feature_names, line))
    return float(click), float(conversion), float(deep_conversion), features
