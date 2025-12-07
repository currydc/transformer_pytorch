import torch
import torch.nn as nn
from positionalencoding import PositionalEncoding

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: 词表大小
            d_model: 词嵌入的维度（通常与Transformer的隐藏层维度一致）
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))


if __name__ == '__main__':
    vocab_size = 100
    d_model = 512
    d_ff = 2048
    embedding = TokenEmbedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (1, 10))
    x=embedding(x)
    pos=PositionalEncoding(d_model)
    out=pos(x)
    print(out.shape)
    print(out)