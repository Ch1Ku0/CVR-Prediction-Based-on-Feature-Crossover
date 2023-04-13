import torch
from torch import nn
from interaction import InnerProductLayer

# 这是多任务学习模型
# 分成了点击、浅层转化、深层转化 三个目标


col_num = 1000
inner_product_layer_num_fields = 12
# 这个EMBEDDING_SIZE 是 FM 层的 embedding层的维度 与右侧的深度模型的embedding层的维度是单独定义的
EMBEDDING_SIZE = 1
tower_dims = [128, 64, 32 + EMBEDDING_SIZE]
dims = [128, 64, 32],
drop_prob = [0.1, 0.3, 0.3]

device = torch.device("cuda:1")
# device = torch.device("cpu")


class Tower(nn.Module):
    def __init__(self,
                 input_dim: int):
        super(Tower, self).__init__()
        self.dims = dims
        self.drop_prob = drop_prob
        self.layer = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                   nn.Dropout(drop_prob[0]),
                                   nn.Linear(128, 64), nn.ReLU(),
                                   nn.Dropout(drop_prob[1]),
                                   nn.Linear(64, 32), nn.ReLU(),
                                   nn.Dropout(drop_prob[2]))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim=32):
        super(Attention, self).__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias=False)
        self.k_layer = nn.Linear(dim, dim, bias=False)
        self.v_layer = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        Q = self.q_layer(inputs)
        K = self.k_layer(inputs)
        V = self.v_layer(inputs)
        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
        a = self.softmax(a)
        outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
        return outputs


class AITM(nn.Module):
    def __init__(self,
                 feature_vocabulary,
                 embedding_size: int):
        super(AITM, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size
        self.embedding_dict = nn.ModuleDict()
        self.__init_weight()

        self.tower_input_size = len(feature_vocabulary) * embedding_size
        self.click_tower = Tower(self.tower_input_size)
        self.conversion_tower = Tower(self.tower_input_size)
        self.deep_conversion_tower = Tower(self.tower_input_size)
        # self.attention_layer = Attention(tower_dims[-1])

        # self.info_layer = nn.Sequential(nn.Linear(tower_dims[-1], tower_dims[-1]), nn.ReLU(),
        #                                 nn.Dropout(drop_prob[-1]))

        self.click_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                         nn.Sigmoid())
        self.conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                              nn.Sigmoid())
        self.deep_conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                              nn.Sigmoid())

        self.fm_layer = FM_Layer(self.feature_vocabulary)



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
        feature_embedding = torch.cat(feature_embedding, 1)

        fm_out = self.fm_layer(x)

        # click任务塔的输出， 输入是embedding
        tower_click = self.click_tower(feature_embedding)

        # conversion任务塔的输出， 输入是embedding
        tower_conversion = self.conversion_tower(feature_embedding)

        # deep_conversion任务塔的输出， 输入是embedding
        tower_deep_conversion = self.deep_conversion_tower(feature_embedding)


        tower_click = torch.cat((tower_click, fm_out), 1)
        tower_conversion = torch.cat((tower_conversion, fm_out), 1)
        tower_deep_conversion = torch.cat((tower_deep_conversion, fm_out), 1)


        # 将click和conversion分别送入各自最后的全连接层和激活层，得到最后的输出结果
        click = torch.squeeze(self.click_layer(tower_click), dim=1)
        conversion = torch.squeeze(self.conversion_layer(tower_conversion), dim=1)
        deep_conversion = torch.squeeze(self.deep_conversion_layer(tower_deep_conversion), dim=1)


        return click, conversion, deep_conversion

    def loss(self,
             click_label,
             click_pred,
             conversion_label,
             conversion_pred,
             deep_conversion_label,
             deep_conversion_pred,
             constraint_weight=0.6):

        click_label = click_label.to(device)
        conversion_label = conversion_label.to(device)
        deep_conversion_label = deep_conversion_label.to(device)

        # click_loss = nn.functional.binary_cross_entropy(click_pred, click_label)
        # conversion_loss = nn.functional.binary_cross_entropy(
        #     conversion_pred, conversion_label)
        #
        # deep_conversion_loss = nn.functional.binary_cross_entropy(
        #     deep_conversion_pred, deep_conversion_label)


        click_loss = self.FL_loss(click_pred, click_label)
        conversion_loss = self.FL_loss(conversion_pred, conversion_label)
        deep_conversion_loss = self.FL_loss(deep_conversion_pred, deep_conversion_label)

        label_constraint = torch.maximum(conversion_pred - click_pred,
                                         torch.zeros_like(click_label))
        constraint_loss = torch.sum(label_constraint)

        deep_label_constraint = torch.maximum(deep_conversion_pred - click_pred,
                                         torch.zeros_like(click_label))
        deep_constraint_loss = torch.sum(deep_label_constraint)

        loss = click_loss + conversion_loss + deep_conversion_loss + constraint_weight * constraint_loss + constraint_weight * deep_constraint_loss
        return loss



class FM_Layer(nn.Module):
    def __init__(self,feature_vocabulary, use_bias=True):
        super(FM_Layer, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.output_activation = nn.Sigmoid()
        self.embedding_dict_fm = nn.ModuleDict()
        self.num_fields = col_num
        # self.inner_product_layer = InnerProductLayer(self.num_fields, output="inner_product")
        self.inner_product_layer = InnerProductLayer(inner_product_layer_num_fields, output="inner_product")
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        self.output_activation = nn.Sigmoid()
        # self.__init_weight_fm()
        self.lr_layer = nn.Sequential(nn.Linear(self.num_fields, EMBEDDING_SIZE),
                                              nn.Sigmoid())

        self.inner_layer = nn.Sequential(nn.Linear(self.num_fields, inner_product_layer_num_fields))

    def __init_weight_fm(self, ):
        for name, size in self.feature_vocabulary.items():
            # emb = nn.Embedding(size, 1)
            emb = nn.Embedding(size, EMBEDDING_SIZE)
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict_fm[name] = emb

    def forward(self, x):

        feature_embedding_fm = []
        for name in self.feature_names:
            # embed = self.embedding_dict_fm[name](x[name])
            embed = torch.unsqueeze(x[name], 1)
            feature_embedding_fm.append(embed)
        feature_embedding = torch.cat(feature_embedding_fm, -1)

        feature_embedding = torch.tensor(feature_embedding, dtype=torch.float32)

        lr_out = self.lr_layer(feature_embedding)
        feature_embedding = self.inner_layer(feature_embedding)
        # if self.bias is not None:
        #     lr_out += self.bias
        # lr_out = self.output_activation(lr_out)
        feature_embedding = feature_embedding.unsqueeze(1)
        dot_sum = self.inner_product_layer(feature_embedding)
        dot_sum = torch.unsqueeze(torch.sum(dot_sum, dim=-1), -1)

        # dot_sum = torch.unsqueeze(dot_sum, -1)
        # dot_sum = dot_sum.repeat(1, lr_out.size()[1], lr_out.size()[2])

        output = dot_sum + lr_out
        # output = self.output_activation(output)
        output = torch.flatten(output, start_dim=1)
        return output


