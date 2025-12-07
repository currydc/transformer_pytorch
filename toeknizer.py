import jieba

def tokenize_chinese(corpus):
    tokenized_corpus = []
    for sentence in corpus:
        tokens = list(jieba.cut(sentence))  # 使用精确模式分词
        tokenized_corpus.append(tokens)
    return tokenized_corpus

corpus = [
    "我喜欢自然语言处理",
    "深度学习让人工智能更智能",
    "我热爱机器学习和深度学习",
    "未来属于人工智能"
]
tokenized_corpus = tokenize_chinese(corpus)
print(tokenized_corpus)


from collections import Counter

def build_vocab(tokenized_corpus, min_freq=1):
    all_tokens = [token for sentence in tokenized_corpus for token in sentence]
    counter = Counter(all_tokens)

    vocab = {'<PAD>': 0, '<UNK>': 1}  # 特殊 token
    current_index = 2
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = current_index
            current_index += 1
    return vocab

vocab = build_vocab(tokenized_corpus)
print(vocab)