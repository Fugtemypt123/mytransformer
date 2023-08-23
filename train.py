import torch
import pickle
import torch.nn as nn
from utils import list2sentence, tensor2sentence, translate_sentence, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from dataset import NumberDataset
from model import Transformer
from sklearn.model_selection import train_test_split


# hyperparameters 1

BATCH_SIZE = 32
EPOCHS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 3e-4
max_len = 120


# get vocabulary

zh_vocab_file_path = "zh_vocab.pkl"
en_vocab_file_path = "en_vocab.pkl"

with open(zh_vocab_file_path, "rb") as zh_vocab_file:
    zh_vocab = pickle.load(zh_vocab_file)

with open(en_vocab_file_path, "rb") as en_vocab_file:
    en_vocab = pickle.load(en_vocab_file)

zh_ivocab = {index: token for token, index in zh_vocab.items()}
en_ivocab = {index: token for token, index in en_vocab.items()}

src_file = 'data/chinese.zh/chinese.zh'
trg_file = 'data/english.en/english.en'


# get dataset

dataset = NumberDataset(src_file, trg_file, zh_vocab, en_vocab, max_len)

'''
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
'''

train_loader =  DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

'''
cnt = 0
for idx, (src, trg) in enumerate(train_loader):
    cnt += 1
    print(src)
    print(trg)
    if cnt == 10:
        break
'''

print(f"--------data loaded--------")

# hyperparameters 2(for model)

src_vocab_size = len(zh_vocab)
trg_vocab_size = len(en_vocab)
embedding_size = 512
num_heads = 8
num_layers = 3
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
forward_expansion = 4
src_pad_idx = 0
trg_pad_idx = 0


# load model

model = Transformer(
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    trg_pad_idx,
    embedding_size,
    num_layers,
    forward_expansion,
    num_heads,
    dropout,
    DEVICE,
    max_len
).to(DEVICE)

'''
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    DEVICE,
).to(DEVICE)
'''

criterion = nn.CrossEntropyLoss(
    ignore_index=trg_pad_idx
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

print(f"--------model loaded--------")


# mode switch

mode = "train"


# start training

print(f"--------start training--------")

# load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)   

for epoch in range(EPOCHS):
    print(f"[Epoch {epoch} / {EPOCHS}]")
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)
    
    model.eval()

    sentence = "经济危机的不断加深使我们看到了危机过后的世界是什么样子的。"

    translated_sentence = translate_sentence(
        model, sentence,  zh_vocab, en_ivocab, DEVICE, max_len=50
    )
    print(translated_sentence)
    model.train()
    losses = []

    for idx, (src, trg) in enumerate(train_loader):
        # src = torch.transpose(src, 0, 1)
        # trg = torch.transpose(trg, 0, 1)

        input = src.to(DEVICE)
        target = trg.to(DEVICE)

        output = model(input, target[:, :-1])

        # print(output.shape)
        # print(target.shape)
        output = output.reshape(-1, output.shape[2])
        target = target[:, 1:].reshape(-1)
        '''
        if idx % 100 == 0:
            # print(input.shape)
            input = input.reshape(-1)
            print(list2sentence(input, zh_ivocab, max_len))
            print(tensor2sentence(output, en_ivocab, max_len))
        '''
        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        if idx % 100 == 0:
            print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch, EPOCHS, idx//100, loss.item()))
    
    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)


    