import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm 

import os
from timeit import default_timer as timer

class Trigrams:
    def __init__(self, all_seqs):
        self.all_seqs = all_seqs
        self.trigram2index = {"SOS": 0, "EOS":1}
        self.index2trigram = {}
        self.trigram2count = {}
        self.num_trigrams = 2 # initially set to 2, (sos, eos)
        
    def addSequence(self, seq): 
        for trigram in seq:
            self.addTrigram(trigram)
    
    def build_vocab(self):
        for i in range(len(self.all_seqs)):
            self.addSequence(self.all_seqs[i])
            
    def addTrigram(self,trigram):
        if trigram not in self.trigram2index:
            self.trigram2index[trigram] = self.num_trigrams
            self.index2trigram[self.num_trigrams] = trigram
            self.trigram2count[trigram] = 1
            self.num_trigrams += 1
        else:
            self.trigram2count[trigram] += 1
            
class NucDataset(Dataset): 
    def __init__(self, datat_dir):
        self.data_dir = datat_dir
        self.sequence_pairs = open(datat_dir, 'r',  encoding='utf8').read().split('\n')
        self.sequence_pairs = [[s.split() for s in l.split('\t')] for l in self.sequence_pairs]
        
        self.src_seqs, self.trg_seqs= self.get_src_trg(self.sequence_pairs)
        
        # self.nucs.build_vocabulary(self.sequence_pairs)
        
        self.src_counter = self.build_counter(self.src_seqs)
        self.trg_counter = self.build_counter(self.trg_seqs)
        
        self.src_vocab = self.build_trigram_vocab(self.src_seqs)
        self.trg_vocab = self.build_trigram_vocab(self.trg_seqs)
        
        
        print('- Number of source sentences: {}'.format(len(self.src_seqs)))
        print('- Number of target sentences: {}'.format(len(self.trg_seqs)))
        print('- Source vocabulary size: {}'.format(len(self.src_vocab.trigram2index)))
        print('- Target vocabulary size: {}'.format(len(self.trg_vocab.trigram2index)))
        
        
    def __len__(self): 
        return len(self.src_seqs)
        
    def build_counter(self, sequences):
        counter = Counter()
        for seq in sequences:
            counter.update(seq)
        return counter
    
    # def build_trigram_vocab(self,counter):
    #     tri_vocab = AttributeDict()
    #     tri_vocab.trigram2index={0:"SOS", 1:"EOS"}
    #     tri_vocab.trigram2index.update({trigram: _id+2 for _id, (trigram, count) in tqdm(enumerate(counter.most_common()))})
    #     tri_vocab.index2trigram = {v:k for k,v in tqdm(tri_vocab.trigram2index.items())}  
    #     return tri_vocab

    def build_trigram_vocab(self, seqs):
        tri_vocab = Trigrams(seqs)
        tri_vocab.build_vocab()
        return tri_vocab
        
    def trigram2indexes(self, trigrams, trigram2index): 
        seq=[]
        seq.append(0)
        seq.extend([trigram2index.get(trigram) for trigram in trigrams])
        seq.append(1)
        return seq
        
    def get_src_trg(self, sequence_pairs):
        
        src_seqs = [] 
        trg_seqs = []
        
        for i in range(len(sequence_pairs)):
            temp = sequence_pairs[i]
            if len(temp)>1:
                src_seqs.append(temp[0])
                trg_seqs.append(temp[1]) 
        return src_seqs, trg_seqs
        
    def __getitem__(self, index):
        src_seq = self.src_seqs[index] 
        trg_seq = self.trg_seqs[index]
        
        src_seq_n = self.trigram2indexes(src_seq, self.src_vocab.trigram2index)
        trg_seq_n = self.trigram2indexes(trg_seq, self.trg_vocab.trigram2index)
        # return src_seq, src_seq_n, trg_seq, trg_seq_n
        return src_seq,torch.tensor(src_seq_n),  trg_seq,torch.tensor(trg_seq_n)

def get_loader(data_dir, batch_size, num_workers, shuffle, pin_memory=True):
    dataset = NucDataset(data_dir)
    loader = DataLoader(    dataset= dataset,
                            batch_size = batch_size,
                            num_workers = num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            collate_fn=collate)
    return loader, dataset


def collate(data): 
    def _pad_sequences(seqs):
        lens=[len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs),max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i,:end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens
        
    src_seqs, src_seq_n, trg_seqs, trg_seq_n=zip(*data)
    src_seq_n, src_lens = _pad_sequences(src_seq_n)
    trg_seq_n, trg_lens = _pad_sequences(trg_seq_n)

    src_seq_n= src_seq_n.transpose(0,1)
    trg_seq_n= trg_seq_n.transpose(0,1)
    
    return src_seq_n, trg_seq_n, src_seqs, trg_seqs, src_lens, trg_lens




DEVICE = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

PAD_IDX =0


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, max_len=1276): 
        super(PositionalEncoding, self).__init__()
        
        # computing the inner part
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000)/embedding_size)
        pos = torch.arange(0,max_len).reshape(max_len,1)
        
        pos_embedding = torch.zeros(max_len, embedding_size)
        # Adding sin and cos at alternate places
        pos_embedding[:,0::2]  = torch.sin(pos * den)
        pos_embedding[:,1::2] = torch.cos(pos * den)
        
        
        '''find the reason for it'''
        pos_embedding = pos_embedding.unsqueeze(-2)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        
    def forward(self, input_embedding):
        # print(input_embedding.size(), self.pos_embedding.size())
        return self.dropout(input_embedding + self.pos_embedding[:input_embedding.size(0),:])
        
        
        
class Mutation_Transformer(nn.Module):
    def __init__(self, n_encoder_layers, n_decoder_layers, embeddings_size, n_head, src_vocab_size, trg_vocab_size, linear_size, dropout): 
        super(Mutation_Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model = embeddings_size, 
                                            nhead = n_head,
                                            num_encoder_layers = n_encoder_layers,
                                            num_decoder_layers = n_decoder_layers,
                                            dim_feedforward = linear_size,
                                            dropout = dropout)
        
        self.generator = nn.Linear(in_features = embeddings_size, out_features = trg_vocab_size)
        self.src_embedding = Embedding(src_vocab_size, embeddings_size)
        self.trg_embedding = Embedding(trg_vocab_size, embeddings_size)
        
        
        self.pos_encoding = PositionalEncoding(embeddings_size, dropout)
        
    def forward(self,src, trg, src_mask, trg_mask, src_pmask, trg_pmask, memory_key_padding_mask):
        src_embedding = self.pos_encoding(self.src_embedding(src))
        trg_embedding = self.pos_encoding(self.trg_embedding(trg))
        
        # print("src : ", src_embedding.size(), src_mask.size(), src_pmask.size(),"targ:", trg_embedding.size(), trg_mask.size(), trg_pmask.size(),"mem", memory_key_padding_mask.size() )

        outputs = self.transformer(src_embedding, trg_embedding, src_mask, trg_mask, None, 
                                src_pmask, trg_pmask, memory_key_padding_mask)
        
        return self.generator(outputs)
    
    def encode(self, src, src_mask):
        return self.transformer.encoder(self.pos_encoding(self.src_embedding(src), src_mask))
    
    def decode(self, trg, trg_mask):
        return self.transformer.encoder(self.positional_encoding(self.trg_embedding(trg), trg_mask))

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size
        
    def forward(self, input): 
        return self.embedding(input.long()) * math.sqrt(self.embedding_size)


'''Understand these two funcitons'''
def generate_square_subsequent_mask(sz): 
    mask = (torch.triu(torch.ones((sz,sz), device = DEVICE))==1).transpose(0,1)
    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))    
    return mask
    
def create_mask(src, trg):
    src_seq_len = src.shape[0]
    trg_seq_len = trg.shape[0]
    
    trg_mask = generate_square_subsequent_mask(trg_seq_len)
    src_mask = generate_square_subsequent_mask(src_seq_len)
    
    src_pmask = (src == PAD_IDX).transpose(0,1)
    trg_pmask = (trg == PAD_IDX).transpose(0,1)
    
    return src_mask, trg_mask, src_pmask, trg_pmask
    

def train_iter(model, optimizer, dataloader):
    
    model.train()
    losses = 0
    
    for i,batch in enumerate(dataloader):
        src,trg,_,_,_,_= batch
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)
        
        
        #explore this line, reducing the dimesnions causes issues, (len-1, batch size) this case (l2n, batch size)
        trg_input = trg[:-1,:]
        # trg_input = trg
        
        # print(src.size(), trg.size(), trg_input.size())

        
        src_mask, trg_mask, src_pmask, trg_pmask = create_mask(src, trg_input)
        
        logits = model(src, trg_input, src_mask, trg_mask, src_pmask, trg_pmask, src_pmask)
        
        
        trg_output=trg[1:,:]
        
        loss = loss_func(logits.reshape(-1, logits.shape[-1]), trg_output.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses+=loss.item()
        
        print(loss.item())
        
    return losses/len(dataloader)
     
def decode(decoding_type, model, src, src_mask, max_len, start_sym):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    
    
    
src_vocab_size = 64 
trg_vocab_size = 65

embeddings_size = 256
n_head = 8
linear_size = 512
n_encoder_layers = 4
n_decoder_layers = 4
dropout = 0.2

batch_size = 2
shuffle = False
pin_memory = True
num_workers = 8

transformer = Mutation_Transformer(n_encoder_layers, n_decoder_layers, embeddings_size, n_head,
                                        src_vocab_size, trg_vocab_size, linear_size, dropout)

for p in transformer.parameters():
    if p.dim()>1: 
        nn.init.xavier_uniform_(p)
transformer = transformer.to(DEVICE)

loss_func = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr = 0.0001, betas = (0.9,0.98), eps = 1e-9)


print("Model Created ... \nAssociating the dataset .....")



data_dir = "/raid/Datasets/aimpid/PairedTextFiles/Nucleotides/ahtisham/Final_w_io.txt"

loader, dataset = get_loader(data_dir, batch_size, num_workers, shuffle, pin_memory)        
        
        
epochs = 5 

for i in range(epochs):
    start_time = timer()
    train_loss = train_iter(transformer, optimizer,loader)
    print(train_loss)
    end_time = timer()  
    

    
