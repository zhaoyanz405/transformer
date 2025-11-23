from transformer.attention import *


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attention, src_attention, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([AddNorm(d_model, dropout) for _ in range(3)])
        self.d_model = d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attention(x, memory, memory, src_mask))
        x = self.sublayers[2](x, self.feed_forward)
        return x