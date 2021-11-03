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

def get_loader(data_obj, batch_size, num_workers, shuffle, pin_memory=True):
    # dataset = NucDataset(data_dir)
    loader = DataLoader(    dataset= data_obj,
                            batch_size = batch_size,
                            num_workers = num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            collate_fn=collate)
    return loader, data_obj


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

    # src_seq_n= src_seq_n.transpose(0,1)
    # trg_seq_n= trg_seq_n.transpose(0,1)
    
    return src_seq_n, trg_seq_n, src_seqs, trg_seqs, src_lens, trg_lens




DEVICE = torch.device('cuda:3' if torch.cuda.is_available else 'cpu')

PAD_IDX =66


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
        # print('dropout',self.dropout(input_embedding + self.pos_embedding[:input_embedding.size(0),:]).size())
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
        
        
        self.pos_encoding = PositionalEncoding(embeddings_size, dropout=dropout)
        
    def forward(self,src, trg, src_mask, trg_mask, src_pmask, trg_pmask, memory_key_padding_mask):
        src_embedding = self.pos_encoding(self.src_embedding(src))
        trg_embedding = self.pos_encoding(self.trg_embedding(trg))
        # print(memory_key_padding_mask)
        
        # print("src : ", src_embedding.size(), src_mask.size(), src_pmask.size(),"targ:", trg_embedding.size(), trg_mask.size(), trg_pmask.size(),"mem", memory_key_padding_mask.size() )

        outputs = self.transformer(src_embedding, trg_embedding, src_mask, trg_mask, None, 
                                src_pmask, trg_pmask, memory_key_padding_mask)
        
        # print(outputs)
        # exit()
        
        return self.generator(outputs)
    
    def encode(self, src, src_mask):
        print('pos_encoding: ',self.pos_encoding(self.src_embedding(src)).size())
        print('srcmask',src_mask.size())
        return self.transformer.encoder(self.pos_encoding(self.src_embedding(src)), src_mask)
    
    def decode(self, trg, memory, trg_mask):
        return self.transformer.decoder(self.pos_encoding(self.trg_embedding(trg)), memory, trg_mask)

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
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    # src_mask = generate_square_subsequent_mask(src_seq_len)
    
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
        optimizer.zero_grad()
        
        # print(src.size())
        # print(trg.size())
        
        #explore this line, reducing the dimesnions causes issues, (len-1, batch size) this case (l2n, batch size)
        trg_input = trg[:-1,:]
        # trg_input = trg
        # print(trg_input.size())
        
        
        src_mask, trg_mask, src_pmask, trg_pmask = create_mask(src, trg_input)
        # print(src_mask.size(), trg_mask.size(), src_pmask.size(), trg_pmask.size())
        
        
        logits = model(src, trg_input, src_mask, trg_mask, 
        src_pmask, trg_pmask, src_pmask)
        # print(logits)
        # exit()
        # print("logits:",logits.size())
        trg_output=trg[1:,:]
        # print(logits.size(), trg_output.size())
        # print("sheikkha ka L",logits.shape[-1], logits.shape)
        
        # print("loss walal L:", logits.reshape(-1, logits.shape[-1]).size(), trg_output.reshape(-1).size())
        
        
        loss = loss_func(logits.reshape(-1, logits.shape[-1]), trg_output.reshape(-1))
        if(i%50 == 0):
            print('loss: ',loss.item())
        
        
        loss.backward()
        optimizer.step()
        losses+=loss.item()
    return losses/len(dataloader)
     
     
def evaluate(model,val_dataloader):
    model.eval()
    losses = 0

    # val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for i, batch in enumerate(val_dataloader):
        src,tgt,_,_,_,_=batch
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        tgt_out = tgt[1:, :]
        loss = loss_func(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)
    
def decode(decoding_type, model, src, src_mask, max_len, start_sym):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    
# function to generate output sequence using greedy algorithm 
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 1:          #'EOS_IDX'
            break   
    return ys

def convert_seq(eval_seq):
    seq_n=[]
    for trigram in eval_seq.split(' '):
        print(trigram)
        seq_n.append(data_obj.src_vocab.trigram2index[trigram])
    print(seq_n)
    return torch.tensor(seq_n)

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = convert_seq(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=0).flatten()
    # return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")    
    return tgt_tokens
    
    

src_vocab_size = 64 
trg_vocab_size = 65

embeddings_size = 512
n_head = 8
linear_size = 512
n_encoder_layers = 3
n_decoder_layers = 3
dropout = 0.2

batch_size = 32
shuffle = False
pin_memory = True
num_workers = 8

transformer = Mutation_Transformer(n_encoder_layers, n_decoder_layers, embeddings_size, n_head,
                                        src_vocab_size, trg_vocab_size, linear_size, dropout)

for p in transformer.parameters():
    if p.dim()>1: 
        nn.init.xavier_uniform_(p)


# if torch.cuda.device_count() > 1:
#     #  print("Let's use", torch.cuda.device_count(), "GPUs!")
#     transformer = torch.nn.DataParallel(transformer)        
        
        
        
        
transformer = transformer.to(DEVICE)

loss_func = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr = 0.0001, betas = (0.9,0.98), eps = 1e-9)


print("Model Created ... \nAssociating the dataset .....")



data_dir = "/raid/Datasets/aimpid/PairedTextFiles/Nucleotides/ahtisham/Final_w_io.txt"
data_obj = NucDataset(data_dir)
loader, full_dataset = get_loader(data_obj, batch_size, num_workers, shuffle, pin_memory)        

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader= DataLoader(    dataset= train_dataset,
                            batch_size = batch_size,
                            num_workers = num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            collate_fn=collate)    
test_loader= DataLoader(    dataset= test_dataset,
                            batch_size = batch_size,
                            num_workers = num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            collate_fn=collate)    



        
epochs = 20

for i in range(epochs):
    start_time = timer()
    train_loss = train_iter(transformer, optimizer,train_loader)
    transformer.eval()
    val_loss = evaluate(transformer, test_loader)
    print("Epoch {}: Training Loss: {}".format((i+1), train_loss))
    print("Epoch {}: Val  Loss: {}".format((i+1), val_loss))
    torch.save(transformer.state_dict(), 'tranformer.pt')
    
    end_time = timer()  

transformer.load_state_dict(torch.load('tranformer.pt'))
transformer.eval()


eval_seq ="ATG TTT GTT TTT CTT GTT TTA TTG CCA CTA GTC TCT AGT CAG TGT GTT AAT CTT ACA ACC AGA ACT CAA TTA CCC CCT GCA TAC ACT AAT TCT TTC ACA CGT GGT GTT TAT TAC CCT GAC AAA GTT TTC AGA TCC TCA GTT TTA CAT TCA ACT CAG GAC TTG TTC TTA CCT TTC TTT TCC AAT GTT ACT TGG TTC CAT GCT ATA CAT GTC TCT GGG ACC AAT GGT ACT AAG AGG TTT GAT AAC CCT GTC CTA CCA TTT AAT GAT GGT GTT TAT TTT GCT TCC ACT GAG AAG TCT AAC ATA ATA AGA GGC TGG ATT TTT GGT ACT ACT TTA GAT TCG AAG ACC CAG TCC CTA CTT ATT GTT AAT AAC GCT ACT AAT GTT GTT ATT AAA GTC TGT GAA TTT CAA TTT TGT AAT GAT CCA TTT TTG GGT GTT TAT TAC CAC AAA AAC AAC AAA AGT TGG ATG GAA AGT GAG TTC AGA GTT TAT TCT AGT GCG AAT AAT TGC ACT TTT GAA TAT GTC TCT CAG CCT TTT CTT ATG GAC CTT GAA GGA AAA CAG GGT AAT TTC AAA AAT CTT AGG GAA TTT GTG TTT AAG AAT ATT GAT GGT TAT TTT AAA ATA TAT TCT AAG CAC ACG CCT ATT AAT TTA GTG CGT GAT CTC CCT CAG GGT TTT TCG GCT TTA GAA CCA TTG GTA GAT TTG CCA ATA GGT ATT AAC ATC ACT AGG TTT CAA ACT TTA CTT GCT TTA CAT AGA AGT TAT TTG ACT CCT GGT GAT TCT TCT TCA GGT TGG ACA GCT GGT GCT GCA GCT TAT TAT GTG GGT TAT CTT CAA CCT AGG ACT TTT CTA TTA AAA TAT AAT GAA AAT GGA ACC ATT ACA GAT GCT GTA GAC TGT GCA CTT GAC CCT CTC TCA GAA ACA AAG TGT ACG TTG AAA TCC TTC ACT GTA GAA AAA GGA ATC TAT CAA ACT TCT AAC TTT AGA GTC CAA CCA ATA GAA TCT ATT GTT AGA TTT CCT AAT ATT ACA AAC TTG TGC CCT TTT GGT GAA GTT TTT AAC GCC ACC AGA TTT GCA TCT GTT TAT GCT TGG AAC AGG AAG AGA ATC AGC AAC TGT GTT GCT GAT TAT TCT GTC CTA TAT AAT TCC GCA TCA TTT TCC ACT TTT AAG TGT TAT GGA GTG TCT CCT ACT AAA TTA AAT GAT CTC TGC TTT ACT AAT GTC TAT GCA GAT TCA TTT GTA ATT AGA GGT GAT GAA GTC AGA CAA ATC GCT CCA GGG CAA ACT GGA AAG ATT GCT GAT TAT AAT TAT AAA TTA CCA GAT GAT TTT ACA GGC TGC GTT ATA GCT TGG AAT TCT AAC AAT CTT GAT TCT AAG GTT GGT GGT AAT TAT AAT TAC CTG TAT AGA TTG TTT AGG AAG TCT AAT CTC AAA CCT TTT GAG AGA GAT ATT TCA ACT GAA ATC TAT CAG GCC GGT AGC ACA CCT TGT AAT GGT GTT GAA GGT TTT AAT TGT TAC TTT CCT TTA CAA TCA TAT GGT TTC CAA CCC ACT AAT GGT GTT GGT TAC CAA CCA TAC AGA GTA GTA GTA CTT TCT TTT GAA CTT CTA CAT GCA CCA GCA ACT GTT TGT GGA CCT AAA AAG TCT ACT AAT TTG GTT AAA AAC AAA TGT GTC AAT TTC AAC TTC AAT GGT TTA ACA GGC ACA GGT GTT CTT ACT GAG TCT AAC AAA AAG TTT CTG CCT TTC CAA CAA TTT GGC AGA GAC ATT GCT GAC ACT ACT GAT GCT GTC CGT GAT CCA CAG ACA CTT GAG ATT CTT GAC ATT ACA CCA TGT TCT TTT GGT GGT GTC AGT GTT ATA ACA CCA GGA ACA AAT ACT TCT AAC CAG GTT GCT GTT CTT TAT CAG GGT GTT AAC TGC ACA GAA GTC CCT GTT GCT ATT CAT GCA GAT CAA CTT ACT CCT ACT TGG CGT GTT TAT TCT ACA GGT TCT AAT GTT TTT CAA ACA CGT GCA GGC TGT TTA ATA GGG GCT GAA CAT GTC AAC AAC TCA TAT GAG TGT GAC ATA CCC ATT GGT GCA GGT ATA TGC GCT AGT TAT CAG ACT CAG ACT AAT TCT CCT CGG CGG GCA CGT AGT GTA GCT AGT CAA TCC ATC ATT GCC TAC ACT ATG TCA CTT GGT GCA GAA AAT TCA GTT GCT TAC TCT AAT AAC TCT ATT GCC ATA CCC ACA AAT TTT ACT ATT AGT GTT ACC ACA GAA ATT CTA CCA GTG TCT ATG ACC AAG ACA TCA GTA GAT TGT ACA ATG TAC ATT TGT GGT GAT TCA ACT GAA TGC AGC AAT CTT TTG TTG CAA TAT GGC AGT TTT TGT ACA CAA TTA AAC CGT GCT TTA ACT GGA ATA GCT GTT GAA CAA GAC AAA AAC ACC CAA GAA GTT TTT GCA CAA GTC AAA CAA ATT TAC AAA ACA CCA CCA ATT AAA GAT TTT GGT GGT TTT AAT TTT TCA CAA ATA TTA CCA GAT CCA TCA AAA CCA AGC AAG AGG TCA TTT ATT GAA GAT CTA CTT TTC AAC AAA GTG ACA CTT GCA GAT GCT GGC TTC ATC AAA CAA TAT GGT GAT TGC CTT GGT GAT ATT GCT GCT AGA GAC CTC ATT TGT GCA CAA AAG TTT AAC GGC CTT ACT GTT TTG CCA CCT TTG CTC ACA GAT GAA ATG ATT GCT CAA TAC ACT TCT GCA CTG TTA GCG GGT ACA ATC ACT TCT GGT TGG ACC TTT GGT GCA GGT GCT GCA TTA CAA ATA CCA TTT GCT ATG CAA ATG GCT TAT AGG TTT AAT GGT ATT GGA GTT ACA CAG AAT GTT CTC TAT GAG AAC CAA AAA TTG ATT GCC AAC CAA TTT AAT AGT GCT ATT GGC AAA ATT CAA GAC TCA CTT TCT TCC ACA GCA AGT GCA CTT GGA AAA CTT CAA GAT GTG GTC AAC CAA AAT GCA CAA GCT TTA AAC ACG CTT GTT AAA CAA CTT AGC TCC AAT TTT GGT GCA ATT TCA AGT GTT TTA AAT GAT ATC CTT TCA CGT CTT GAC AAA GTT GAG GCT GAA GTG CAA ATT GAT AGG TTG ATC ACA GGC AGA CTT CAA AGT TTG CAG ACA TAT GTG ACT CAA CAA TTA ATT AGA GCT GCA GAA ATC AGA GCT TCT GCT AAT CTT GCT GCT ACT AAA ATG TCA GAG TGT GTA CTT GGA CAA TCA AAA AGA GTT GAT TTT TGT GGA AAG GGC TAT CAT CTT ATG TCC TTC CCT CAG TCA GCA CCT CAT GGT GTA GTC TTC TTG CAT GTG ACT TAT GTC CCT GCA CAA GAA AAG AAC TTC ACA ACT GCT CCT GCC ATT TGT CAT GAT GGA AAA GCA CAC TTT CCT CGT GAA GGT GTT TTT GTT TCA AAT GGC ACA CAC TGG TTT GTA ACA CAA AGG AAT TTT TAT GAA CCA CAA ATC ATT ACT ACA GAC AAC ACA TTT GTG TCT GGT AAC TGT GAT GTT GTA ATA GGA ATT GTC AAC AAC ACA GTT TAT GAT CCT TTG CAA CCT GAA TTA GAC TCA TTC AAG GAG GAG TTA GAT AAA TAT TTT AAG AAT CAT ACA TCA CCA GAT GTT GAT TTA GGT GAC ATC TCT GGC ATT AAT GCT TCA GTT GTA AAC ATT CAA AAA GAA ATT GAC CGC CTC AAT GAG GTT GCC AAG AAT TTA AAT GAA TCT CTC ATC GAT CTC CAA GAA CTT GGA AAG TAT GAG CAG TAT ATA AAA TGG CCA TGG TAC ATT TGG CTA GGT TTT ATA GCT GGC TTG ATT GCC ATA GTA ATG GTG ACA ATT ATG CTT TGC TGT ATG ACC AGT TGC TGT AGT TGT CTC AAG GGC TGT TGT TCT TGT GGA TCC TGC TGC AAA TTT GAT GAA GAC GAC TCT GAG CCA GTG CTC AAA GGA GTC AAA TTA CAT"#TAC ACA TAA
translated=translate(transformer,eval_seq)

    
print("CV",convert_seq(eval_seq))
print('TT',translated)