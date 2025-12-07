import torch
import torch.nn as nn
from positionalencoding import PositionalEncoding
from embedding import TokenEmbedding
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


if __name__ == '__main__':
    vocab_size = 100
    d_model = 512
    d_ff = 2048
    embedding = TokenEmbedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (1, 10))
    x = embedding(x)
    pos = PositionalEncoding(d_model)
    x = pos(x)
    feedforward = FeedForward(d_model, d_ff)
    x=feedforward(x)
    print(x.shape)
    print(x)