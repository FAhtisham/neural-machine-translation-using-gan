from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm 




class Trigrams:
    def __init__(self, name):
        self.name = name 
        self.trigram2index = {}
        self.index2trigram = {0: "SOS", 1: "EOS"}
        self.trigram2count = {}
        self.num_trigrams = 2 # initially set to 2, (sos, eos)
        
    def addSequence(self, seq): 
        for trigram in seq:
            self.addTrigram(trigram)
            
    def addTrigram(self,trigram):
        if trigram not in self.trigram2index:
            self.trigram2index[trigram] = self.num_trigrams
            self.index2trigram[self.num_trigrams] = trigram
            self.trigram2count[trigram] = 1
            self.num_trigrams += 1
        else:
            self.trigram2count[trigram] += 1

class AttributeDict(dict):
    
    def __init__(self, *av, **kv):
        dict.__init__(self, *av, **kv)
        self.__dict__=self

class NucDataset(Dataset): 
    def __init__(self, datat_dir):
        self.data_dir = datat_dir
        self.sequence_pairs = open(datat_dir, 'r', encoding='utf-8').read().split('\n')
        self.sequence_pairs = [[s.split() for s in l.split('\t')] for l in self.sequence_pairs]
        
        self.src_seqs, self.trg_seqs= self.get_src_trg(self.sequence_pairs)
        
        # self.nucs.build_vocabulary(self.sequence_pairs)
        
        self.src_counter = self.build_counter(self.src_seqs)
        self.trg_counter = self.build_counter(self.trg_seqs)
        
        self.src_vocab = self.build_trigram_vocab(self.src_counter)
        self.trg_vocab = self.build_trigram_vocab(self.trg_counter)
        
        
        print('- Number of source sentences: {}'.format(len(self.src_seqs)))
        print('- Number of target sentences: {}'.format(len(self.trg_seqs)))
        print('- Source vocabulary size: {}'.format(len(self.src_vocab.trigram2index)))
        print('- Target vocabulary size: {}'.format(len(self.trg_vocab.trigram2index)))
        print('- Target vocabulary size: {}'.format(self.trg_vocab.trigram2index))
        
        
        
    def build_counter(self, sequences):
        counter = Counter()
        for seq in sequences:
            counter.update(seq)
        return counter
    
    def build_trigram_vocab(self,counter):
        tri_vocab = AttributeDict()
        tri_vocab.trigram2index={0:"SOS", 1:"EOS"}
        tri_vocab.trigram2index.update({trigram: _id+2 for _id, (trigram, count) in tqdm(enumerate(counter.most_common()))})
        tri_vocab.index2trigram = {v:k for k,v in tqdm(tri_vocab.trigram2index.items())}  
        return tri_vocab

    def trigram2indexes(self, trigrams, trigram2index): 
        seq=[]
        seq.append(SOS)
        seq.extend([trigram2index.get(trigram) for trigram in trigrams])
        seq.append(EOS)
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
        return src_seq, src_seq_n, trg_seq, trg_seq_n
        



obj_ = NucDataset("/raid/Datasets/aimpid/PairedTextFiles/Nucleotides/ahtisham/Final_w_io.txt")

print(len(obj_.src_seqs), len(obj_.trg_seqs))
        
        
        
        
        
        
        