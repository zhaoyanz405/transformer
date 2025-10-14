from torch import nn

from utils import *
from encoder import EncoderLayer
from decoder import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_embed = nn.Sequential(EmbeddingLayer(src_vocab, d_model), PositionalEncoding(d_model, dropout))
        self.tgt_embed = nn.Sequential(EmbeddingLayer(tgt_vocab, d_model), PositionalEncoding(d_model, dropout))

        attention = MultiHeadAttention(h, d_model, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, attention, feed_forward, dropout) for _ in range(N)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, attention, attention, feed_forward, dropout) for _ in range(N)])
        self.out = nn.Linear(d_model, tgt_vocab)

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.tgt_embed(tgt)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        x = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.out(x)