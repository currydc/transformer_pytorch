
parallel_data = [
    ("I love NLP", "我喜欢自然语言处理"),
    ("Deep learning is powerful", "深度学习很强大"),
    ("Machine learning is fun", "机器学习很有趣"),
    ("NLP is cool", "自然语言处理很酷"),
    ("I love you", "我喜欢你"),
    ("who are you","你是谁"),
    ("Private spaceflight companies now offer not just a short trip to near space but the tantalizing possibility of living our lives on Mars.",
     "私人太空旅行公司现在不仅仅 提供近太空的短程旅行， 还提供了可以在火星上 居住的诱人的可能性。"),
    ("Glaciers and sea ice that have been with us for millennia are now disappearing in a matter of decades.",
     "冰川和海上的浮冰跟我们一起相处了几千年，但在过去几十年的时间里却正在逐渐消失。"),
    ("Kepler's data reveals planets' sizes as well as their distance from their parent star.",
     "开普勒望远镜的数据揭示了行星的大小，以及它们与其母恒星之间的距离。")
]

def tokenize_en(sentence):
    return sentence.lower().split()

import jieba

def tokenize_zh(sentence):
    return list(jieba.cut(sentence))

from collections import Counter

en_tokens = []
zh_tokens = []

for en_sent, zh_sent in parallel_data:
    en_tokens.extend(tokenize_en(en_sent))
    zh_tokens.extend(tokenize_zh(zh_sent))


def build_vocab(tokens, min_freq=1, special_tokens=['<PAD>', '<UNK>', '<SOS>', '<EOS>']):
    counter = Counter(tokens)

    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    current_index = len(special_tokens)

    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = current_index
            current_index += 1

    return vocab


def encode(sentence, vocab, lang='en'):
    if lang == 'en':
        tokens = tokenize_en(sentence)
    else:
        tokens = tokenize_zh(sentence)

    ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    ids = [vocab['<SOS>']] + ids + [vocab['<EOS>']]
    return ids

if __name__ == '__main__':
    en_vocab = build_vocab(en_tokens)
    zh_vocab = build_vocab(zh_tokens)
    print(en_vocab)
    print(zh_vocab)
    encoded_en = encode("I love deep learning", en_vocab, lang='en')
    encoded_zh = encode("机器学习很有趣", zh_vocab, lang='zh')
    print(encoded_en)
    print(encoded_zh)