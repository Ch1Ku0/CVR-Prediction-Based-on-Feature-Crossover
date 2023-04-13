import torch
from torch import nn
from interaction import InnerProductLayer

# 这是多任务学习模型
# 分成了点击、转化转化 两个目标
# 使用了fmFM模型

col_num = 18
# col_num = 5
task_num = 2

device = torch.device("cuda:1")


# device = torch.device("cpu")


class PLE(nn.Module):
    def __init__(self,
                 feature_vocabulary,
                 embedding_size: int):
        super(PLE, self).__init__()

        # self.embed_dim = 128
        self.bottom_mlp_dims = (512, 256)
        self.tower_mlp_dims = (128, 64)
        self.task_num = task_num
        self.shared_expert_num = 4
        self.specific_expert_num = 4
        self.dropout = 0.2

        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size
        self.embedding_dict = nn.ModuleDict()
        self.__init_weight()

        self.embed_output_dim = col_num * self.embedding_size
        self.layers_num = len(self.bottom_mlp_dims)

        self.expert = list()
        for i in range(self.layers_num):
            if i == 0:
                self.expert.append(torch.nn.ModuleList(
                    [MultiLayerPerceptron(self.embed_output_dim, [self.bottom_mlp_dims[i]], self.dropout,
                                          output_layer=False) for
                     j in range(self.specific_expert_num * self.task_num + self.shared_expert_num)]))
            else:
                self.expert.append(torch.nn.ModuleList(
                    [MultiLayerPerceptron(self.bottom_mlp_dims[i - 1], [self.bottom_mlp_dims[i]], self.dropout,
                                          output_layer=False) for
                     j in range(self.specific_expert_num * self.task_num + self.shared_expert_num)]))
        self.expert = torch.nn.ModuleList(self.expert)

        self.gate = list()
        for i in range(self.layers_num):
            if i == 0:
                input_dim = self.embed_output_dim
            else:
                input_dim = self.bottom_mlp_dims[i - 1]
            gate_list = [
                torch.nn.Sequential(torch.nn.Linear(input_dim, self.shared_expert_num + self.specific_expert_num),
                                    torch.nn.Softmax(dim=1)) for j in range(self.task_num)]
            gate_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, self.shared_expert_num + self.task_num * self.specific_expert_num),
                    torch.nn.Softmax(dim=1)))
            self.gate.append(torch.nn.ModuleList(gate_list))
        self.gate = torch.nn.ModuleList(self.gate)

        # 索引按任务数量区分，在这个配置下，每个experts中的expert的数量是4
        # 任务A expert索引是 0 1 2 3 8 9 10 11
        # 任务B expert索引是 4 5 6 7 8 9 10 11
        # 8 9 10 11 是共享expert
        self.task_expert_index = list()
        for i in range(self.task_num):
            index_list = list()
            index_list.extend(range(i * self.specific_expert_num, (1 + i) * self.specific_expert_num))
            index_list.extend(range(self.task_num * self.specific_expert_num,
                                    self.task_num * self.specific_expert_num + self.shared_expert_num))
            self.task_expert_index.append(index_list)
        self.task_expert_index.append(range(self.task_num * self.specific_expert_num + self.shared_expert_num))

        self.tower = torch.nn.ModuleList(
            [MultiLayerPerceptron(self.bottom_mlp_dims[-1], self.tower_mlp_dims, self.dropout) for i in
             range(self.task_num)])

    def __init_weight(self, ):
        for name, size in self.feature_vocabulary.items():
            emb = nn.Embedding(size, self.embedding_size)
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict[name] = emb

    def forward(self, x):
        feature_embedding = []
        for name in self.feature_names:
            embed = self.embedding_dict[name](x[name])
            feature_embedding.append(embed)
        feature_embedding = torch.cat(feature_embedding, 1).view(-1, self.embed_output_dim)


        # 两个任务各有一个experts， 还有一个共享的experts，输入都是相同的
        task_fea = [feature_embedding for i in range(self.task_num + 1)]
        for i in range(self.layers_num):
            for j in range(self.task_num + 1):
                # 输入experts，输出一个结果
                fea = torch.cat(
                    [self.expert[i][index](task_fea[j]).unsqueeze(1) for index in self.task_expert_index[j]], dim=1)
                # 输入gate， 输出一个结果
                gate_value = self.gate[i][j](task_fea[j]).unsqueeze(1)
                # torch.bmm 计算两个矩阵的乘法
                task_fea[j] = torch.bmm(gate_value, fea).squeeze(1)

        # 在这里输入进tower中的只有task A 和 B 的experts，共享的experts不参与计算了
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        click = results[0]
        conversion = results[1]


        return click, conversion

    def loss(self,
             click_label,
             click_pred,
             conversion_label,
             conversion_pred,
             constraint_weight=0.6):
        # device="cpu"):
        click_label = click_label.to(device)
        conversion_label = conversion_label.to(device)

        click_loss = nn.functional.binary_cross_entropy(click_pred, click_label)
        conversion_loss = nn.functional.binary_cross_entropy(
            conversion_pred, conversion_label)

        label_constraint = torch.maximum(conversion_pred - click_pred,
                                         torch.zeros_like(click_label))
        constraint_loss = torch.sum(label_constraint)

        loss = click_loss + conversion_loss + constraint_weight * constraint_loss
        return loss


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


