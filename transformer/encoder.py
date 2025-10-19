from utils import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, sekf_attention, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = sekf_attention
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([AddNorm(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  