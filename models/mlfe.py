import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from models.MLFE.embed import FP_embedding
from models.MLFE.tools import Transpose

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=None, activation=F.relu, batch_first=True):
        super(Encoder, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,  #
                                                  dropout=dropout,
                                                  activation=activation,
                                                  batch_first=batch_first
                                                  )

        self.norm_ttn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

    def forward(self, data):
        b = data.shape[0]
        src = self.encoder(data)
        src = self.norm_ttn(src)
        return src


class MsEncoderBasic(nn.Module):
    def __init__(self, args, **factory_kwargs):
        super(MsEncoderBasic, self).__init__()

        self.seq_length = getattr(args, "mask_freq", 20000)
        self.pred_len = 0
        self.d_model = args.d_model
        self.num_en_layers = args.num_layers
        self.pred_len = args.nclass
        self.num_globalfeature = args.nfeature
        dropout = args.dropout
        self.nhead = 8
        self.output_attention = False
        self.seg_len = getattr(args, 'seg_len', 500)
        self.n_statistic_patch = getattr(args, 'n_statistic_patch', 200)    # default 200

        init_seed = getattr(args, "init_seed", None)
        torch.manual_seed(init_seed)
        print(f"torch randon int {torch.randint(100000, (1,))}")

        # add encoding
        self.en_encoding = FP_embedding(self.seg_len, self.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, (self.seq_length // self.seg_len), self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        # add encoder
        en_layers = OrderedDict()
        for i in range(self.num_en_layers):
            en_layers[f"encoder_layer_{i}"] = Encoder(
                d_model=self.d_model,
                nhead=self.nhead,
                activation=F.relu
            )
        self.en_layers = nn.Sequential(en_layers)
        self.en_ln = nn.LayerNorm(self.d_model)
        # add projection header for classification
        self.projection_header_method()
        # add statistic process method
        self.statistic_header_method()
        # init weights
        self.init_weights()

        self.sigmoid = nn.Sigmoid()

    def projection_header_method(self):
        heads_layers = OrderedDict()
        heads_layers["decoder_header"] = nn.Linear(self.d_model, self.pred_len)
        heads_layers["detection_flatten"] = nn.Flatten(start_dim=-2)
        heads_layers["layer_norm"] = nn.LayerNorm(self.seq_length // self.seg_len)
        heads_layers["detection_header"] = nn.Linear(self.seq_length // self.seg_len, self.pred_len)
        heads_layers["dropout"] = nn.Dropout(0.)
        self.heads = nn.Sequential(heads_layers)

    def statistic_header_method(self):
        pass

    def init_weights(self):
        for layer in self.heads:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal(layer.weight)

    def forward(self, data):
        src = self.get_src_data(data)
        src = self.en_encoding(src)
        src += self.enc_pos_embedding
        src = self.pre_norm(src)
        for encoder in self.en_layers:
            src = encoder(src)
        src = self.en_ln(src)
        res = self.heads(src)

        feature = self.get_feature_data(data)

        res = self.process_feature_data(res, feature)

        if self.output_attention:
            return res, None
        else:
            return res

    def get_src_data(self, data):
        fft_feature = data["batch_x_maskedfft"]
        torch._assert(fft_feature.dim() == 2, f"src: Expected (batch_size, seq_length) got {fft_feature.shape}")
        src = fft_feature[:, :self.seq_length]
        return src

    def get_feature_data(self, data):
        pass

    def process_feature_data(self, res, feature):
        res = self.sigmoid(res)
        return res

    def get_features(self):
        return getattr(self, 'features', None)


class MsEncoderwFeature(MsEncoderBasic):
    def statistic_header_method(self):
        self.global_feature = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.n_statistic_patch * 9, self.n_statistic_patch * 9 // 2),
            nn.LayerNorm(self.n_statistic_patch * 9 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_statistic_patch * 9 // 2, self.pred_len),
        )
        self.fusion_linear = nn.Linear(2, self.pred_len)

    def init_weights(self):
        for layer in self.heads:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal(layer.weight)

        for layer in self.global_feature:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal(layer.weight)

    def get_feature_data(self, data):
        feature = data["batch_x_statisticalfeatures"]
        torch._assert(feature.dim() == 3, f"src: Expected (batch_size, feature_num, seq_length) got {feature.shape}")
        return feature

    def process_feature_data(self, res, feature):
        res_global = self.global_feature(feature).reshape(-1, 1)
        self.features = torch.hstack((res, res_global))
        res = self.sigmoid(self.fusion_linear(torch.hstack((res, res_global))))
        return res


class MsEncoderwCluster(MsEncoderwFeature):
    def get_feature_data(self, data):
        feature = data["batch_x_statisticalfeatures"]
        ClusterFeatures = data["batch_x_clusterfeatures"]
        torch._assert(feature.dim() == 3, f"src: Expected (batch_size, feature_num, seq_length) got {feature.shape}")
        feature = torch.concat((feature, ClusterFeatures.transpose(2, 1)), dim=1)
        return feature

