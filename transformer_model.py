import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PositionalEncoding(nn.Module):
    """
    位置编码层，为输入序列添加位置信息
    d_model：嵌入维度,每个token会被映射为一个d_model维的向量
    max_len：表示 位置编码（Positional Encoding）所能支持的最大序列长度 ，
    如果你设置 max_len=5000，那么模型可以处理最长为 5000 的序列。
    如果你的输入序列长度是 100，那么就会从预先生成的 pe 中取前 100 个位置编码加到输入上
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """

    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k])).to(device)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性变换并分割成多头
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 应用mask（如果有）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context = torch.matmul(attn_weights, v)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出线性变换
        output = self.w_o(context)

        return output, attn_weights


class PositionwiseFeedforward(nn.Module):
    """
    位置前馈网络
    d_ff:隐藏层的维度大小,通常是d_model的4倍
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    """

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, src_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈网络子层
        ff_output = self.feedforward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    """

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 交叉注意力子层
        attn_output, _ = self.cross_attn(x , enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # 前馈网络子层
        ff_output = self.feedforward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型（编码器-解码器结构）
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()

        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # 编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.linear = nn.Linear(d_model, tgt_vocab_size)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encoder(self, src, src_mask=None):
        """单独编码方法"""
        src_embedded = self.positional_encoding(self.src_embedding(src))
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        return enc_output
    def decoder(self, dec_output,enc_output,src_mask, trg_mask=None):
        """单独解码方法"""
        tgt_embedded = self.positional_encoding(self.tgt_embedding(dec_output))
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, trg_mask)
        return dec_output
    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # 编码器部分
        src_embedded = self.positional_encoding(self.src_embedding(src))
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # 解码器部分
        tgt_embedded = self.positional_encoding(self.tgt_embedding(trg))
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, trg_mask)

        # 输出层
        output = self.linear(dec_output)

        return output


# 示例使用
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    d_model = 512
    n_head = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    dropout = 0.1
    batch_size = 32
    seq_len = 50

    # 创建模型
    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, n_head,
        num_encoder_layers, num_decoder_layers, d_ff, dropout
    ).to(device)

    # 创建示例输入
    src = torch.randint(0, src_vocab_size, (batch_size, seq_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len)).to(device)

    # 创建mask（简单示例，实际应用中需要更复杂的mask逻辑）
    src_mask = torch.ones((batch_size, 1, 1, seq_len)).to(device)
    tgt_mask = torch.tril(torch.ones((seq_len, seq_len))).expand(batch_size, 1, seq_len, seq_len).to(device)

    # 前向传播
    output = model(src, tgt, src_mask, tgt_mask)

    print(f"Input shape: src={src.shape}, tgt={tgt.shape}")
    print(f"Output shape: {output.shape}")
