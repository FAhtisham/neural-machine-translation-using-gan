# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import math

# from torch.utils.data import Dataset, DataLoader
# from collections import Counter
# from tqdm import tqdm 

# import os
# from timeit import default_timer as timer

# class Trigrams:
#     def __init__(self, all_seqs):
#         self.all_seqs = all_seqs
#         self.trigram2index = {"SOS": 0, "EOS":1}
#         self.index2trigram = {0:"SOS", 1:"EOS"}
#         self.trigram2count = {}
#         self.num_trigrams = 2 # initially set to 2, (sos, eos)
        
#     def addSequence(self, seq): 
#         for trigram in seq:
#             self.addTrigram(trigram)
    
#     def build_vocab(self):
#         for i in range(len(self.all_seqs)):
#             self.addSequence(self.all_seqs[i])
            
#     def addTrigram(self,trigram):
#         if trigram not in self.trigram2index:
#             self.trigram2index[trigram] = self.num_trigrams
#             self.index2trigram[self.num_trigrams] = trigram
#             self.trigram2count[trigram] = 1
#             self.num_trigrams += 1
#         else:
#             self.trigram2count[trigram] += 1
            
# class NucDataset(Dataset): 
#     def __init__(self, datat_dir):
#         self.data_dir = datat_dir
#         self.sequence_pairs = open(datat_dir, 'r',  encoding='utf8').read().split('\n')
#         self.sequence_pairs = [[s.split() for s in l.split('\t')] for l in self.sequence_pairs]
        
#         self.src_seqs, self.trg_seqs= self.get_src_trg(self.sequence_pairs)
        
#         # self.nucs.build_vocabulary(self.sequence_pairs)
        
#         self.src_counter = self.build_counter(self.src_seqs)
#         self.trg_counter = self.build_counter(self.trg_seqs)
        
#         self.src_vocab = self.build_trigram_vocab(self.src_seqs)
#         self.trg_vocab = self.build_trigram_vocab(self.trg_seqs)
        
        
#         print('- Number of source sentences: {}'.format(len(self.src_seqs)))
#         print('- Number of target sentences: {}'.format(len(self.trg_seqs)))
#         print('- Source vocabulary size: {}'.format(len(self.src_vocab.trigram2index)))
#         print('- Target vocabulary size: {}'.format(len(self.trg_vocab.trigram2index)))
        
        
#     def __len__(self): 
#         return len(self.src_seqs)
        
#     def build_counter(self, sequences):
#         counter = Counter()
#         for seq in sequences:
#             counter.update(seq)
#         return counter
    
#     # def build_trigram_vocab(self,counter):
#     #     tri_vocab = AttributeDict()
#     #     tri_vocab.trigram2index={0:"SOS", 1:"EOS"}
#     #     tri_vocab.trigram2index.update({trigram: _id+2 for _id, (trigram, count) in tqdm(enumerate(counter.most_common()))})
#     #     tri_vocab.index2trigram = {v:k for k,v in tqdm(tri_vocab.trigram2index.items())}  
#     #     return tri_vocab

#     def build_trigram_vocab(self, seqs):
#         tri_vocab = Trigrams(seqs)
#         tri_vocab.build_vocab()
#         return tri_vocab
        
#     def trigram2indexes(self, trigrams, trigram2index): 
#         seq=[]
#         seq.append(0)
#         seq.extend([trigram2index.get(trigram) for trigram in trigrams])
#         seq.append(1)
#         return seq
        
#     def get_src_trg(self, sequence_pairs):
        
#         src_seqs = [] 
#         trg_seqs = []
        
#         for i in range(len(sequence_pairs)):
#             temp = sequence_pairs[i]
#             if len(temp)>1:
#                 src_seqs.append(temp[0])
#                 trg_seqs.append(temp[1]) 
#         return src_seqs, trg_seqs
        
#     def __getitem__(self, index):
#         src_seq = self.src_seqs[index] 
#         trg_seq = self.trg_seqs[index]
        
#         src_seq_n = self.trigram2indexes(src_seq, self.src_vocab.trigram2index)
#         trg_seq_n = self.trigram2indexes(trg_seq, self.trg_vocab.trigram2index)
#         sample = {"src": src_seq_n, "trg": trg_seq_n}

#         # return src_seq, src_seq_n, trg_seq, trg_seq_n
#         # return src_seq,torch.tensor(src_seq_n),  trg_seq,torch.tensor(trg_seq_n)
#         return sample

            
# def get_loader(data_obj, batch_size, num_workers, shuffle, pin_memory=True):
#     # dataset = NucDataset(data_dir)
#     loader = DataLoader(    dataset= data_obj,
#                             batch_size = batch_size,
#                             num_workers = num_workers,
#                             shuffle=shuffle,
#                             pin_memory=pin_memory,
#                             collate_fn=None)
#     return loader, data_obj


# def collate(data): 
#     # def _pad_sequences(seqs):
#     #     lens=[len(seq) for seq in seqs]
#     #     padded_seqs = torch.zeros(len(seqs),max(lens)).long()
#     #     for i, seq in enumerate(seqs):
#     #         end = lens[i]
#     #         padded_seqs[i,:end] = torch.LongTensor(seq[:end])
#     #     return padded_seqs, lens
#     print(list(data))     
#     src_seqs, src_seq_n, trg_seqs, trg_seq_n=zip(data) # removed *
    
#     exit()
#     # src_seq_n, src_lens = _pad_sequences(src_seq_n)
#     # trg_seq_n, trg_lens = _pad_sequences(trg_seq_n)

#     # src_seq_n= src_seq_n.transpose(0,1)
#     # trg_seq_n= trg_seq_n.transpose(0,1)
    
#     print("test")
#     print(type(src_seq_n), type(trg_seq_n))
#     src_seq_n = torch.tensor(src_seq_n)
#     trg_seq_n=torch.tensor(trg_seq_n)
#     return src_seq_n, trg_seq_n, src_seqs, trg_seqs




#______________________________________________________________________________________________________________________________________



from utils import indexesFromSequence
from utils import *

import random
import torch 

class Trigrams:
    def __init__(self, name):
        self.name = name 
        self.trigram2index = {}
        self.index2trigram = {0: "SOS", 1: "EOS"}
        self.trigram2count = {}
        self.num_trigrams = 2 # initially set to 2, (sos, eos)
        
        
    def addSequence(self, seq): 
        #.split(' '): 
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

def prepareData(lang1, lang2, reverse = False):
    input_lang, output_lang, pairs = readTrigrams(lang1, lang2, reverse)
    print("Read sequence pairs : ", len(pairs))
    
    #pairs = filterPairs(pairs)
    print("Counting words")

    for i in range(1,len(pairs)-1):
        input_lang.addSequence(pairs[i][0])
        output_lang.addSequence(pairs[i][1]) 
    
    print("Trigrams : ")
    print("L1 Trigrams: ",input_lang.name, input_lang.num_trigrams)
    print("L2 Trigrams: ",output_lang.name, output_lang.num_trigrams) 
    return input_lang, output_lang, pairs




def readTrigrams(lang1, lang2, reverse = False): 
    print("Reading Files . . . . ")

    lines = open('/raid/Datasets/aimpid/PairedTextFiles/Nucleotides/ahtisham/Final_w_io.txt', encoding = 'utf-8').read().split('\n')
    # split the sequences into pairs 
    pairs = [[normalizeString2(s) for s in l.split('\t')] for l in lines]

    
    # reverse pairs, scnsjdncjsdang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs] 
        input_lang = Trigrams(lang1)
        output_lang = Trigrams(lang2)
        
    else:
        
        input_lang = Trigrams(lang1) 
        output_lang = Trigrams(lang2)
    
    return input_lang, output_lang, pairs
    
    
def get_gfloat_vals(input_lang, output_lang,pairs):
    src_seqs, trg_seqs=[],[]
    for pair in pairs:
        src_seqs.append(indexesFromSequence(input_lang, pair[0]))
        trg_seqs.append(indexesFromSequence(output_lang, pair[1]))
        
    src_seqs = torch.LongTensor(src_seqs)#.to(torch.int64)    
    trg_seqs = torch.LongTensor(trg_seqs)#.to(torch.int64)
    
    return src_seqs, trg_seqs 
    
# def random_batch(batch_size, input_lang, output_lang, pairs): 
#     input_seqs = []
#     target_seqs = [] 
    
#     # choose random pairs 
#     for i in range(batch_size): 
#         pair = random.choice(pairs)
        
#         input_seqs.append(indexesFromSequence(input_lang,pair[0]))
#         target_seqs.append(indexesFromSequence(output_lang,pair[1]))
        
        
        
#         '''
#             An alternative approach can be to use the torch.nn.util.rnn.pad(), this function can automatically add padding into any sequence
#             Here we write our own code to do the padding
#         '''
        
        
#     # sorting in a descending manner on the basis of the lengths of seqs    
#     seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
#     input_seqs, target_seqs = zip(*seq_pairs)
    
#     # apply the pad here
#     input_lengths = [len(s) for s in input_seqs]
#     max_length_i = max(input_lengths) 
#     input_padded = [addPadding(s, max_length_i) for s in input_seqs]    
    
#     target_lengths = [len(s) for s in target_seqs]
#     max_length_t = max(target_lengths)
#     target_padded = [addPadding(s, max_length_t) for s in target_seqs]
    
#     # Padded arrays into (Batch * max_length) tensors and modify them to (max_length * batch)
#     input_tensors = torch.LongTensor(input_padded).transpose(0,1)
#     target_tensors = torch.LongTensor(target_padded).transpose(0,1)
    
#     input_tensors = input_tensors
#     target_tensors = target_tensors
#     # return input_tensors.size(), target_tensors.size()

    return input_tensors, input_lengths, target_tensors, target_lengths
    
    
    
    
    
    
    
    
    
    
    
        
        
    
    