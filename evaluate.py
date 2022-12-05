import Levenshtein
import transrate
import readfile
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
import torch
import torch.nn as nn

model_dir_path = transrate.model_dir_path
model_name = transrate.model_name

texts_src_test = transrate.texts_src_test
texts_tgt_test = transrate.texts_tgt_test
device = transrate.device
tokenizer_src = transrate.tokenizer_src
tokenizer_tgt = transrate.tokenizer_tgt

vocab_src = transrate.vocab_src
vocab_tgt = transrate.vocab_tgt

PAD_IDX = transrate.PAD_IDX
START_IDX = transrate.START_IDX
END_IDX = transrate.END_IDX

train_data = transrate.train_data

best_model = transrate.model
best_model.load_state_dict(torch.load(model_dir_path.joinpath(model_name)))
evaltxt = 'LevenSrc.txt'

def translate(
    model, text, vocab_src, vocab_tgt, tokenizer_src, seq_len_tgt,
    START_IDX, END_IDX
):
    
    model.eval()
    tokens = transrate.convert_text_to_indexes(text=text, vocab=vocab_src, tokenizer=tokenizer_src)
    num_tokens = len(tokens)
    src = torch.LongTensor(tokens).reshape(num_tokens, 1)
    mask_src = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    
    predicts = greedy_decode(
        model=model, src=src,
        mask_src=mask_src, seq_len_tgt=seq_len_tgt,
        START_IDX=START_IDX, END_IDX=END_IDX
    ).flatten()
    
    return ' '.join([vocab_tgt.get_itos()[token] for token in predicts]).replace("<start>", "").replace("<end>", "")


def greedy_decode(model, src, mask_src, seq_len_tgt, START_IDX, END_IDX):
    
    src = src.to(device)
    mask_src = mask_src.to(device)

    memory = model.encode(src, mask_src)
    memory = model.transformer_encoder(model.positional_encoding(model.token_embedding_src(src)), mask_src)
    ys = torch.ones(1, 1).fill_(START_IDX).type(torch.long).to(device)
    
    for i in range(seq_len_tgt - 1):
        
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        mask_tgt = (transrate.generate_square_subsequent_mask(ys.size(0),PAD_IDX).type(torch.bool)).to(device)
        
        output = model.decode(ys, memory, mask_tgt)
        output = output.transpose(0, 1)
        output = model.output(output[:, -1])
        _, next_word = torch.max(output, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == END_IDX:
            break
            
    return ys

seq_len_tgt = max([len(x[1]) for x in transrate.train_data])

sum_ratio = 0
total_test = 0

fl = open(evaltxt, 'w')

for i in tqdm(range(len(texts_src_test))):
    mbyte = translate(
        model=best_model, text=texts_src_test[i], vocab_src=vocab_src, vocab_tgt=vocab_tgt,
        tokenizer_src=tokenizer_src, seq_len_tgt=seq_len_tgt,
        START_IDX=START_IDX, END_IDX=END_IDX
    )
    total_test += 1
    Lratio = Levenshtein.ratio(texts_tgt_test[i],mbyte)
    sum_ratio += Lratio
    fl.write("{}\n".format(Lratio))

fl.close()
print(total_test)
print(sum_ratio/total_test)