import torch
import spacy


def translate_sentence(model, sentence,  zh_vocab, en_ivocab, device, max_len):
    spacy_zh = spacy.load('zh_core_web_sm')
    # spacy_en = spacy.load('en_core_web_sm')

    zh_tokens = [tok.text for tok in spacy_zh(sentence)]
    # en_tokens = [tok.text for tok in spacy_en(answer)]

    zh_nums = []
    # en_nums = []

    for word in zh_tokens:
        try:
            zh_nums.append(zh_vocab[word])
        except KeyError:
            pass
        
    zh_nums.insert(0, 1)
    zh_nums.append(2)

    # zh_res = [0] * max_len
    # zh_res[:len(zh_nums)] = zh_nums
    '''
    for word in en_tokens:
        try:
            en_nums.append(en_vocab[word])
        except KeyError:
            pass

    en_nums.insert(0, 1)
    en_nums.append(2)
        
    en_res = [0] * max_len
    en_res[:len(en_nums)] = en_nums
    '''
    src = torch.tensor(zh_nums).unsqueeze(1).to(device)
    src = torch.transpose(src, 0, 1)
    '''
    trg = torch.tensor(en_res).unsqueeze(1).to(device)
    trg = torch.transpose(trg, 0, 1)

    output = model(src, trg)
    output = output.reshape(-1, output.shape[2])
    return tensor2sentence(output, en_ivocab, max_len)

    '''
    # outputs = [0] * max_len
    outputs = [1]

    for i in range(max_len):
        trg = torch.tensor(outputs).unsqueeze(1).to(device)
        trg = torch.transpose(trg, 0, 1)

        # print(src.shape)
        # print(trg.shape)

        with torch.no_grad():
            output = model(src, trg)

        # print(output.shape)

        best_guess = output.argmax(2)[:, -1].item()
        # print(best_guess)
        outputs.append(best_guess)

        # print(outputs[i+1])

        if best_guess == 2:
            break
    
    
    translated_sentence = ""

    for i in range(max_len):
        translated_sentence += en_ivocab[outputs[i]]
        if outputs[i] == 2 or outputs[i] == 0:
            break
        else:
            translated_sentence += " "

    return translated_sentence


def tensor2sentence(output, en_ivocab, max_len):
    best_words = output.argmax(1)
    sentence = ""
    for i in range(max_len):
        num = best_words[i].item()
        word = en_ivocab[num]
        sentence += word
        if word != "<eos>":
            sentence += " "
        else:
            break
    return sentence

def list2sentence(best_words, zh_ivocab, max_len):
    sentence = ""
    for i in range(max_len):
        num = best_words[i].item()
        word = zh_ivocab[num]
        sentence += word
        if word != "<eos>":
            sentence += " "
        else:
            break
    return sentence

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
