from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm 


import torch

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
    
    
#################################
data_dir = "/raid/Datasets/aimpid/PairedTextFiles/Nucleotides/ahtisham/Final_w_io.txt"
batch_size = 32
shuffle = False
pin_memory = True
num_workers = 8
loader, dataset = get_loader(data_dir, batch_size, num_workers, shuffle, pin_memory)        
        
        
for i in range(1):
    print("Chala")
    for i, batch in enumerate(loader):
        print(i)
        # batch =  batch.cuda()
        x,y,_,_,_,_ = batch
        # x,y=batch
        print(type(x), type(y))

        
        