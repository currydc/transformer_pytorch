import torch.nn as nn
import torch
from positionalencoding import PositionalEncoding

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dim_feedforward, max_seq_length=100, dropout=0.1):
        """
        Args:
            input_dim: 输入词汇表大小或特征维度（如果是连续输入）
            embed_dim: 嵌入向量维度
            num_heads: 多头注意力的头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络的中间维度
            max_seq_length: 最大序列长度
            dropout: dropout率
        """
        super(TransformerEncoderModel, self).__init__()

        # 嵌入层（适用于NLP任务，如文本分类）
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # 位置编码（Positional Encoding）
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_seq_length)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 输入形状为 [batch_size, seq_len, embed_dim]
        )

        # Transformer 编码器（多层）
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层（可选，用于下游任务如分类）
        self.fc_out = nn.Linear(embed_dim, input_dim)

    def forward(self, src):
        """
        Args:
            src: 输入张量形状 [batch_size, seq_len]，元素是token索引
        Returns:
            输出张量形状 [batch_size, seq_len, embed_dim]
        """
        src = self.embedding(src)  # [batch_size, seq_len, embed_dim]
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src)  # [batch_size, seq_len, embed_dim]
        logits = self.fc_out(memory)
        return memory, logits