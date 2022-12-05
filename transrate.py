from pathlib import Path
import math
import time
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import (
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
import readfile
from transformers import BertModel
import Levenshtein

model_dir_path = Path('translateModel')
model_name = 'gpu_translationSrc_transfomer.pth'

batch_size = 1

embedding_size = 256
nhead = 8
dim_feedforward = 100
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.1

def build_vocab(texts, tokenizer):
    
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    return vocab(counter, specials=['<unk>', '<pad>', '<start>', '<end>'])

def data_process(texts_src, texts_tgt, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt):
    
    data = []
    for (src, tgt) in zip(texts_src, texts_tgt):
        src_tensor = torch.tensor(
            convert_text_to_indexes(text=src, vocab=vocab_src, tokenizer=tokenizer_src),
            dtype=torch.long
        )
        tgt_tensor = torch.tensor(
            convert_text_to_indexes(text=tgt, vocab=vocab_tgt, tokenizer=tokenizer_tgt),
            dtype=torch.long
        )
        data.append((src_tensor, tgt_tensor))
        
    return data

def convert_text_to_indexes(text, vocab, tokenizer):
    return [vocab['<start>']] + [vocab[token] for token in tokenizer(text.strip("\n"))] + [vocab['<end>']]

def generate_batch(data_batch):
    
    batch_src, batch_tgt = [], []
    for src, tgt in data_batch:
        batch_src.append(src)
        batch_tgt.append(tgt)
        
    batch_src = pad_sequence(batch_src, padding_value=PAD_IDX)
    batch_tgt = pad_sequence(batch_tgt, padding_value=PAD_IDX)
    
    return batch_src, batch_tgt

class Seq2SeqTransformer(nn.Module):
    
    def __init__(
        self, num_encoder_layers: int, num_decoder_layers: int,
        embedding_size: int, vocab_size_src: int, vocab_size_tgt: int,
        dim_feedforward:int = 512, dropout:float = 0.1, nhead:int = 8
    ):
        
        super(Seq2SeqTransformer, self).__init__()

        self.token_embedding_src = TokenEmbedding(vocab_size_src, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.token_embedding_tgt = TokenEmbedding(vocab_size_tgt, embedding_size)
        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output = nn.Linear(embedding_size, vocab_size_tgt)

    def forward(
        self, src: Tensor, tgt: Tensor,
        mask_src: Tensor, mask_tgt: Tensor,
        padding_mask_src: Tensor, padding_mask_tgt: Tensor,
        memory_key_padding_mask: Tensor
    ):
        
        embedding_src = self.positional_encoding(self.token_embedding_src(src))
        memory = self.transformer_encoder(embedding_src, mask_src, padding_mask_src)
        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(
            embedding_tgt, memory, mask_tgt, None,
            padding_mask_tgt, memory_key_padding_mask
        )
        return self.output(outs)

    def encode(self, src: Tensor, mask_src: Tensor):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt: Tensor, memory: Tensor, mask_tgt: Tensor):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)

class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size
        
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)

class PositionalEncoding(nn.Module):
    
    def __init__(self, embedding_size: int, dropout: float, maxlen: int = 6000):
        super(PositionalEncoding, self).__init__()
        
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        embedding_pos = torch.zeros((maxlen, embedding_size))
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den)
        embedding_pos = embedding_pos.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.embedding_pos[: token_embedding.size(0), :])

def create_mask(src, tgt, PAD_IDX):
    
    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt,PAD_IDX)
    mask_src = torch.zeros((seq_len_src, seq_len_src), device=device).type(torch.bool)

    padding_mask_src = (src == PAD_IDX).transpose(0, 1)
    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1)
    
    return mask_src, mask_tgt, padding_mask_src, padding_mask_tgt


def generate_square_subsequent_mask(seq_len, PAD_IDX):
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == PAD_IDX, float(0.0))
    return mask

def train(model, data, optimizer, criterion, PAD_IDX):
    
    model.train()
    losses = 0
    for src, tgt in tqdm(data):
        
        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, :]

        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)

        logits = model(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )

        optimizer.zero_grad()

        output_tgt = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), output_tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        
    return losses / len(data)


def evaluate(model, data, criterion, PAD_IDX):
    
    model.eval()
    losses = 0
    for src, tgt in data:
        
        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, :]

        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)

        logits = model(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )
        
        output_tgt = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), output_tgt.reshape(-1))
        losses += loss.item()
        
    return losses / len(data)

class JavaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def greedy_decode(model, src, mask_src, seq_len_tgt, START_IDX, END_IDX):
    
    src = src.to(device)
    mask_src = mask_src.to(device)

    memory = model.encode(src, mask_src)
    memory = model.transformer_encoder(model.positional_encoding(model.token_embedding_src(src)), mask_src)
    ys = torch.ones(1, 1).fill_(START_IDX).type(torch.long).to(device)
    
    for i in range(seq_len_tgt - 1):
        
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        mask_tgt = (generate_square_subsequent_mask(ys.size(0),PAD_IDX).type(torch.bool)).to(device)
        
        output = model.decode(ys, memory, mask_tgt)
        output = output.transpose(0, 1)
        output = model.output(output[:, -1])
        _, next_word = torch.max(output, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == END_IDX:
            break
            
    return ys

def translate(
    model, text, vocab_src, vocab_tgt, tokenizer_src, seq_len_tgt,
    START_IDX, END_IDX
):
    
    model.eval()
    tokens = convert_text_to_indexes(text=text, vocab=vocab_src, tokenizer=tokenizer_src)
    num_tokens = len(tokens)
    src = torch.LongTensor(tokens).reshape(num_tokens, 1)
    mask_src = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    
    predicts = greedy_decode(
        model=model, src=src,
        mask_src=mask_src, seq_len_tgt=seq_len_tgt,
        START_IDX=START_IDX, END_IDX=END_IDX
    ).flatten()
    
    return ' '.join([vocab_tgt.get_itos()[token] for token in predicts]).replace("<start>", "").replace("<end>", "")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')



if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)

#datadivide = 16021

texts_src, texts_tgt = readfile.readf(0,1000)
datadivide = len(texts_src)//10

texts_src_train = texts_src[:datadivide*8]
texts_src_valid = texts_src[datadivide*8:datadivide*9]
texts_src_test = texts_src[datadivide*9:]

texts_tgt_train = texts_tgt[:datadivide*8]
texts_tgt_valid = texts_tgt[datadivide*8:datadivide*9]
texts_tgt_test = texts_tgt[datadivide*9:]

print("trainsize:",len(texts_src_train))
print("validsize:",len(texts_src_valid))
print("testsize:",len(texts_src_test))
#texts_src_train, texts_tgt_train = readfile.readf(0,datadivide)
#texts_src_valid, texts_tgt_valid = readfile.readf(datadivide,35000)

#for src, tgt in zip(texts_src_train[:3], texts_tgt_train[:3]):
#    print(src.strip('\n'))
#    print('↓')
#    print(tgt.strip('\n'))
#    print('')
tokenizer_src = get_tokenizer(readfile.srctokenize)
tokenizer_tgt = get_tokenizer(readfile.texttokenize)

vocab_src = build_vocab(texts_src_train, tokenizer_src)
vocab_tgt = build_vocab(texts_tgt_train, tokenizer_tgt)

# for char, index in list(vocab_src.get_stoi().items())[:15]:
#     print('Char: {: <8} → Index: {: <2}'.format(char, index))

vocab_src.set_default_index(vocab_src['<unk>'])
vocab_tgt.set_default_index(vocab_tgt['<unk>'])



train_data = data_process(
    texts_src=texts_src_train, texts_tgt=texts_tgt_train,
    vocab_src=vocab_src, vocab_tgt=vocab_tgt,
    tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt
)
valid_data = data_process(
    texts_src=texts_src_valid, texts_tgt=texts_tgt_valid,
    vocab_src=vocab_src, vocab_tgt=vocab_tgt,
    tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt
)

PAD_IDX = vocab_src['<pad>']
START_IDX = vocab_src['<start>']
END_IDX = vocab_src['<end>']



train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)



vocab_size_src = len(vocab_src)
vocab_size_tgt = len(vocab_tgt)

model = Seq2SeqTransformer(
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    embedding_size=embedding_size,
    vocab_size_src=vocab_size_src, vocab_size_tgt=vocab_size_tgt,
    dim_feedforward=dim_feedforward,
    dropout=dropout, nhead=nhead
)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# model = BertModel.from_pretrained("bert-base-multilingual-cased")

model = model.to(device)


if __name__=='__main__':

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(model.parameters())

    epoch = 100
    best_loss = float('Inf')
    best_model = None
    patience = 10
    counter = 0

    time_sta = time.time()
    for loop in range(1, epoch + 1):
        
        start_time = time.time()
        
        loss_train = train(
            model=model, data=train_iter, optimizer=optimizer,
            criterion=criterion, PAD_IDX=PAD_IDX
        )
        
        elapsed_time = time.time() - start_time
        
        loss_valid = evaluate(
            model=model, data=valid_iter, criterion=criterion, PAD_IDX=PAD_IDX
        )
        
        print('[{}/{}] train loss: {:.2f}, valid loss: {:.2f}  [{}{:.0f}s] count: {}, {}'.format(
            loop, epoch,
            loss_train, loss_valid,
            str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
            elapsed_time % 60,
            counter,
            '**' if best_loss > loss_valid else ''
        ))
        
        if best_loss > loss_valid:
            best_loss = loss_valid
            best_model = model
            counter = 0
            
        if counter > patience:
            break
        
        counter += 1
    # 時間計測終了
    time_end = time.time()
    # 経過時間（秒）
    tim = time_end- time_sta

    print(tim)
    torch.save(best_model.state_dict(), model_dir_path.joinpath(model_name))

    # seq_len_tgt = max([len(x[1]) for x in train_data])

    # sum_ratio = 0
    # total_test = 0

    # fl = open('LevenSrc.txt', 'w')

    # for i in tqdm(range(len(texts_src_test))):
    #     try:
            
    #         mbyte = translate(
    #             model=best_model, text=texts_src_test[i], vocab_src=vocab_src, vocab_tgt=vocab_tgt,
    #             tokenizer_src=tokenizer_src, seq_len_tgt=seq_len_tgt,
    #             START_IDX=START_IDX, END_IDX=END_IDX
    #         )
    #         total_test += 1
    #         Lratio = Levenshtein.ratio(texts_tgt_test[i],mbyte)
    #         sum_ratio += Lratio
    #         fl.write("{}\n".format(Lratio))
    #     except:
    #             pass

    # fl.close()
    # print(total_test)
    # print(sum_ratio/total_test)