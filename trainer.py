import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from vocabulary import *
from transformer_model import Transformer
from torch.utils.tensorboard import SummaryWriter

# 给定的词汇表
src_vocab = build_vocab(en_tokens)
trg_vocab = build_vocab(zh_tokens)
# src_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
#              'i': 4, 'love': 5, 'nlp': 6, 'deep': 7,
#              'learning': 8, 'is': 9, 'powerful': 10}
#
# trg_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
#              '我': 4, '喜欢': 5, '自然语言': 6, '处理': 7,
#              '深度': 8, '学习': 9, '很': 10, '强大': 11}

# 反转词汇表（索引到token）
src_itos = {v: k for k, v in src_vocab.items()}
trg_itos = {v: k for k, v in trg_vocab.items()}

# 示例数据
train_data = parallel_data


# 创建数据集类
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, trg_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, trg_text = self.data[idx]

        # 将文本转换为索引序列
        # src_tokens = src_text.split()
        # trg_tokens = trg_text.split()
        src_tokens=tokenize_en(src_text)
        trg_tokens=tokenize_zh(trg_text)

        # 数值化并添加起止标记
        src_indices = [self.src_vocab['<SOS>']] + \
                      [self.src_vocab.get(token, self.src_vocab['<UNK>']) for token in src_tokens] + \
                      [self.src_vocab['<EOS>']]

        trg_indices = [self.trg_vocab['<SOS>']] + \
                      [self.trg_vocab.get(token, self.trg_vocab['<UNK>']) for token in trg_tokens] + \
                      [self.trg_vocab['<EOS>']]

        return {
            "src": torch.tensor(src_indices, dtype=torch.long),
            "trg": torch.tensor(trg_indices, dtype=torch.long),
            "src_len": len(src_indices),
            "trg_len": len(trg_indices)
        }


# 数据加载器的collate函数
def collate_fn(batch):
    src_seqs = [item["src"] for item in batch]
    trg_seqs = [item["trg"] for item in batch]
    src_lens = [item["src_len"] for item in batch]
    trg_lens = [item["trg_len"] for item in batch]

    # 填充到批次内最大长度,同一个 batch 中的所有样本必须具有相同的形状（shape）(堆叠成同一个tensor)
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, padding_value=0, batch_first=True)
    trg_padded = nn.utils.rnn.pad_sequence(trg_seqs, padding_value=0, batch_first=True)

    return {
        "src": src_padded,
        "trg": trg_padded,
        "src_lens": torch.tensor(src_lens),
        "trg_lens": torch.tensor(trg_lens)
    }


# 创建数据集和数据加载器
dataset = TranslationDataset(train_data, src_vocab, trg_vocab)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
pass

writer=SummaryWriter('runs')
def train_transformer(
        model,
        train_loader,
        optimizer,
        criterion,
        src_vocab,
        trg_vocab,
        n_epochs=10,
        clip=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            src = batch["src"].to(device)  # [batch_size, src_len]
            trg = batch["trg"].to(device)  # [batch_size, trg_len]

            # 创建mask
            src_mask = (src != src_vocab['<PAD>']).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]

            trg_pad_mask = (trg != trg_vocab['<PAD>']).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, trg_len]
            trg_len = trg.size(1)
            trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()  # [trg_len, trg_len]
            trg_mask = trg_pad_mask & trg_sub_mask  # [batch_size, 1, trg_len, trg_len]

            optimizer.zero_grad()

            # 前向传播 (使用teacher forcing)
            output = model(
                src=src,
                trg=trg[:, :-1],  # 去掉eos token
                src_mask=src_mask,
                trg_mask=trg_mask[:, :, :-1, :-1]  # 调整mask大小
            )  # [batch_size, trg_len-1, output_dim]
            # 计算损失
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)  # [batch_size*(trg_len-1), output_dim]
            trg = trg[:, 1:].contiguous().view(-1)  # [batch_size*(trg_len-1)]

            loss = criterion(output, trg)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{n_epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        # 启动tensorboard
        # tensorboard - -logdir =./ Transformer / runs
        print(f"Epoch [{epoch + 1}/{n_epochs}] | Avg Loss: {avg_loss:.4f}")
        # torch.save(model.state_dict(), 'transformer-model_2.pt')
        from safetensors.torch import save_file
        save_file(model.state_dict(), 'transformer-model_2.safetensors')
    print("Training completed!")
    writer.close()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(trg_vocab),
    d_model=128,  # 减小模型尺寸以适应小数据集
    n_head=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=512,
    dropout=0.1
).to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab['<PAD>'])

# 训练模型
trained_model = train_transformer(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    n_epochs=200,
    clip=1.0,
    device=device
)