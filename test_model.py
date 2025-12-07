from transformer_model import *
import torch
from vocabulary import *
# 给定的词汇表
# src_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
#              'i': 4, 'love': 5, 'nlp': 6, 'deep': 7,
#              'learning': 8, 'is': 9, 'powerful': 10}
#
# trg_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
#              '我': 4, '喜欢': 5, '自然语言': 6, '处理': 7,
#              '深度': 8, '学习': 9, '很': 10, '强大': 11}
src_vocab = build_vocab(en_tokens)
trg_vocab = build_vocab(zh_tokens)

# 反转词汇表（索引到token）
src_itos = {v: k for k, v in src_vocab.items()}
trg_itos = {v: k for k, v in trg_vocab.items()}
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
        # output=F.softmax(output, dim=-1)

        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(
    src_vocab_size=len(src_vocab), tgt_vocab_size=len(trg_vocab), d_model=128, n_head=4,
    num_encoder_layers=2, num_decoder_layers=2, d_ff=512, dropout=0.1
).to(device)
from safetensors.torch import load_file
state_dict = load_file("transformer-model_2.safetensors")
model.load_state_dict(state_dict)
weight=model.state_dict()
keys=weight.keys()
model.eval()
print("模型加载完成！")



# 文本预处理函数
def preprocess_sentence(sentence, vocab):
    tokens = sentence.lower().split()
    tokens = ['<SOS>'] + tokens + ['<EOS>']
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return torch.LongTensor(indices).unsqueeze(0).to(device)  # 添加batch维度

# 测试数据
# test_sentences = [
#     "I love NLP",
#     "Deep learning is powerful",
#     "Machine learning is fun",
#     "NLP is cool",
#     "I like Machine"
# ]

test_sentences =[data[0] for data in parallel_data]

# 准备测试数据
test_data = [preprocess_sentence(sent, src_vocab) for sent in test_sentences]
pass

def translate_batch(
        model,
        src_sequences,
        src_vocab,
        trg_vocab,
        trg_itos,
        device,
        max_len=100
):
    translations = []

    for src in src_sequences:
        # 创建源mask
        src_mask = (src != src_vocab['<PAD>']).unsqueeze(1).unsqueeze(2)

        # 编码器前向传播
        with torch.no_grad():
            enc_output = model.encoder(src, src_mask)

        # 初始化目标序列
        trg_indices = [trg_vocab['<SOS>']]

        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)

            # 创建目标mask
            trg_pad_mask = (trg_tensor != trg_vocab['<PAD>']).unsqueeze(1).unsqueeze(2)
            trg_len = trg_tensor.size(1)
            trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
            trg_mask = trg_pad_mask & trg_sub_mask

            # 解码器前向传播
            with torch.no_grad():
                output = model.decoder(trg_tensor, enc_output, src_mask, trg_mask)
                output = model.linear(output)

            # 获取预测的下一个token
            pred_token_idx = output.argmax(2)[:, -1].item()
            trg_indices.append(pred_token_idx)

            # 遇到EOS停止
            if pred_token_idx == trg_vocab['<EOS>']:
                break

        # 将索引转换为token
        trg_tokens = [trg_itos[idx] for idx in trg_indices[1:-1]]  # 去掉<SOS>和<EOS>
        translations.append(' '.join(trg_tokens))

    return translations


# 执行翻译
translations = translate_batch(
    model=model,
    src_sequences=test_data,
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    trg_itos=trg_itos,
    device=device
)

# 显示结果
print("\n测试结果：")
print("-" * 50)
for src, trans in zip(test_sentences, translations):
    print(f"源文: {src}")
    print(f"翻译: {trans}")
    print("-" * 50)