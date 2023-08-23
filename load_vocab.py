import pickle

vocab_file_path = "vocab.pkl"

with open(vocab_file_path, "rb") as vocab_file:
    loaded_vocab = pickle.load(vocab_file)

print(len(loaded_vocab))