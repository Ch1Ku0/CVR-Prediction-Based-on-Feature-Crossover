from numpy import product
import torch
from torch import nn

tower_dims = [128, 64, 32]
dims = [128, 64, 32]
drop_prob = [0.1, 0.3, 0.3]


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Tower(nn.Module):
    def __init__(self,
                 input_dim: int):
        super(Tower, self).__init__()
        self.dims = dims
        self.drop_prob = drop_prob
        self.layer = nn.Sequential(nn.Linear(input_dim, dims[0]), nn.ReLU(),
                                   nn.Dropout(drop_prob[0]),
                                   nn.Linear(dims[0], dims[1]), nn.ReLU(),
                                   nn.Dropout(drop_prob[1]),
                                   nn.Linear(dims[1], dims[2]), nn.ReLU(),
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
        print(inputs.size())
        Q = self.q_layer(inputs)
        K = self.k_layer(inputs)
        V = self.v_layer(inputs)
        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
        print(a.size())
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
        self.attention_layer = Attention(tower_dims[-1])

        self.info_layer = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(),
                                        nn.Dropout(drop_prob[-1]))

        self.click_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                         nn.Sigmoid())
        self.conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                              nn.Sigmoid())

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
        tower_click = self.click_tower(feature_embedding)

        tower_conversion = torch.unsqueeze(
            self.conversion_tower(feature_embedding), 1)

        info = torch.unsqueeze(self.info_layer(tower_click), 1)

        print(tower_conversion.size())
        print(info.size())
        ait = self.attention_layer(torch.cat([tower_conversion, info], 1))
        print(ait.size())
        click = torch.squeeze(self.click_layer(tower_click), dim=1)
        conversion = torch.squeeze(self.conversion_layer(ait), dim=1)

        return click, conversion

    def loss(self,
             click_label,
             click_pred,
             conversion_label,
             conversion_pred,
             constraint_weight=0.6):
        click_label = click_label.to(device)
        conversion_label = conversion_label.to(device)

        click_loss = nn.functional.binary_cross_entropy(
            click_pred, click_label)
        conversion_loss = nn.functional.binary_cross_entropy(
            conversion_pred, conversion_label)

        label_constraint = torch.maximum(conversion_pred - click_pred,
                                         torch.zeros_like(click_label))
        constraint_loss = torch.sum(label_constraint)

        loss = click_loss + conversion_loss + constraint_weight * constraint_loss
        return loss
