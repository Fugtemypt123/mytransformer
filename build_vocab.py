import pickle
import itertools
from collections import Counter
from dataset import WordDataset

zh_vocab_file_path = "zh_vocab.pkl"
en_vocab_file_path = "en_vocab.pkl"
src_file = 'data/chinese.zh/chinese.zh'
trg_file = 'data/english.en/english.en'

dataset = WordDataset(src_file, trg_file)
zh_tokens = list(itertools.chain.from_iterable(src_tokens for src_tokens, trg_tokens in dataset))
en_tokens = list(itertools.chain.from_iterable(trg_tokens for src_tokens, trg_tokens in dataset))
zh_vocab_counter = Counter(zh_tokens)
en_vocab_counter = Counter(en_tokens)

MIN_FREQ = 2
all_vocab = {
    '<pad>':0,
    '<sos>':1,
    '<eos>':2,
    '<unk>':3
}
zh_vocab = {
    '<pad>':0,
    '<sos>':1,
    '<eos>':2,
    '<unk>':3
}
en_vocab = {
    '<pad>':0,
    '<sos>':1,
    '<eos>':2,
    '<unk>':3
}

zh_tot = 4
for index, (token, freq) in enumerate(zh_vocab_counter.items()):
    if freq >= MIN_FREQ:
        zh_vocab.update({token: zh_tot})
        zh_tot += 1    

en_tot = 4
for index, (token, freq) in enumerate(en_vocab_counter.items()):
    if freq >= MIN_FREQ:
        en_vocab.update({token: en_tot})
        en_tot += 1  

print(f"zh_vocab_len = {len(zh_vocab)}")
print(f"en_vocab_len = {len(en_vocab)}")

with open(zh_vocab_file_path, "wb") as zh_vocab_file:
    pickle.dump(zh_vocab, zh_vocab_file)

with open(en_vocab_file_path, "wb") as en_vocab_file:
    pickle.dump(en_vocab, en_vocab_file)