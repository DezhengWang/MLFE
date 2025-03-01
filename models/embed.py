import torch.nn as nn
from einops import rearrange


class FP_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(FP_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) -> (b seg_num) seg_len', seg_len=self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b seg_num) d_model -> b seg_num d_model', b=batch)

        return x_embed
