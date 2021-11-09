import argparse
import options
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda


import math

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm 

import os
from timeit import default_timer as timer
from collections import OrderedDict

from Dataset import *
from models import LSTMModel, Discriminator
from PGLoss import * 
from meters import *
import arguments
import utils

use_cuda = True 

#####################################################################################
##################### Defining the Variables ########################################
#####################################################################################

DEVICE = torch.device("cuda:3" if torch.cuda.is_available else "cpu")



args = argparse.ArgumentParser(description="Mutation GAN NMT")
args = arguments.add_generator_model_args(args)
args = arguments.add_discriminator_model_args(args)
args = arguments.add_optimization_args(args)
args = arguments.add_general_args(args)
args = args.parse_args()




logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
g_logging_meters = OrderedDict()
g_logging_meters['train_loss'] = AverageMeter()
g_logging_meters['valid_loss'] = AverageMeter()
g_logging_meters['train_acc'] = AverageMeter()
g_logging_meters['valid_acc'] = AverageMeter()
g_logging_meters['bsz'] = AverageMeter()  # sentences per batch

d_logging_meters = OrderedDict()
d_logging_meters['train_loss'] = AverageMeter()
d_logging_meters['valid_loss'] = AverageMeter()
d_logging_meters['train_acc'] = AverageMeter()
d_logging_meters['valid_acc'] = AverageMeter()
d_logging_meters['bsz'] = AverageMeter()  # sentences per batch


args.encoder_embed_dim = 256
args.encoder_layers = 2 # 4
args.encoder_dropout_out = 0
args.decoder_embed_dim = 1000
args.decoder_layers = 2 # 4
args.decoder_out_embed_dim = 1000
args.decoder_dropout_out = 0
args.bidirectional = False

data_dir = "/raid/Datasets/aimpid/PairedTextFiles/Nucleotides/ahtisham/Final_w_io.txt"

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

src_vocab_size = 64 
trg_vocab_size = 65

print("Loading Dataset ....")
data_obj = NucDataset(data_dir)

print("Dataset Loaded ....")
print(args.encoder_embed_dim, args.encoder_layers)

print(len(data_obj.src_vocab.index2trigram))

generator = LSTMModel(args, data_obj.src_vocab.index2trigram, data_obj.trg_vocab.index2trigram,  use_cuda=True)
print(generator)

discriminator = Discriminator(args, data_obj.src_vocab.index2trigram, data_obj.trg_vocab.index2trigram, use_cuda=True)
print(discriminator)





if use_cuda:
    if torch.cuda.device_count() > 1:
        discriminator = torch.nn.DataParallel(discriminator).cuda()
        generator = torch.nn.DataParallel(generator).cuda()
    else:
        generator.cuda()
        discriminator.cuda()
else:
    discriminator.cpu()
    generator.cpu()





if not os.path.exists('checkpoints/joint'):
    os.makedirs('checkpoints/joint')
checkpoints_path = 'checkpoints/joint/'



###########################
# define loss function
g_criterion = torch.nn.NLLLoss( reduction='sum')
d_criterion = torch.nn.BCELoss()
pg_criterion = PGLoss( size_average=True,reduce=True)

# # fix discriminator word embedding (as Wu et al. do)
# for p in discriminator.module.embed_src_tokens.parameters():
#     p.requires_grad = False
# for p in discriminator.module.embed_trg_tokens.parameters():
#     p.requires_grad = False

# define optimizer
g_optimizer = eval("torch.optim." + args.g_optimizer)(filter(lambda x: x.requires_grad,
                                                            generator.parameters()),
                                                    args.g_learning_rate)

d_optimizer = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad,
                                                            discriminator.parameters()),
                                                    args.d_learning_rate,
                                                    momentum=args.momentum,
                                                    nesterov=True)




best_dev_loss = math.inf
num_update = 0

loader, data_obj= get_loader(data_obj, batch_size, num_workers, shuffle)
    

generator.train()
for i, sample in enumerate(loader):
    # _,src,_,trg= sample
    # print("a",src.size(), trg.size())
    # print(data_obj.src_vocab.index2trigram)
    # src = src.cuda()
    # trg = trg.cuda()
    # if use_cuda:
    # # wrap input tensors in cuda tensors
    sample = utils.make_variable(sample, cuda=cuda)
    # ## part I: use gradient policy method to train the generator

    # # use policy gradient training when random.random() > 50%
    if random.random()  >= 0.5:
        print(sample)
        print("Policy Gradient Training")
        sys_out_batch = generator(sample) # 64 X 50 X 6632
    else:
        print("simple training")

exit()
exit()



    # max_positions_train = (args.fixed_max_len, args.fixed_max_len)

    # # Initialize dataloader, starting at batch_offset
    # trainloader = dataset.train_dataloader(
    #     'train',
    #     max_tokens=args.max_tokens,
    #     max_sentences=args.joint_batch_size,
    #     max_positions=max_positions_train,
    #     # seed=seed,
    #     epoch=epoch_i,
    #     sample_without_replacement=args.sample_without_replacement,
    #     sort_by_source_size=(epoch_i <= args.curriculum),
    #     shard_id=args.distributed_rank,
    #     num_shards=args.distributed_world_size,
    # )

# main training loop
for epoch_i in range(1, args.epochs + 1):
    logging.info("At {0}-th epoch.".format(epoch_i))

    seed = args.seed + epoch_i
    torch.manual_seed(seed)


    # reset meters
    for key, val in g_logging_meters.items():
        if val is not None:
            val.reset()
    for key, val in d_logging_meters.items():
        if val is not None:
            val.reset()

    # set training mode
    generator.train()
    discriminator.train()
    # update_learning_rate(num_update, 8e4, args.g_learning_rate, args.lr_shrink, g_optimizer)

    for i, sample in enumerate(trainloader):
    
        if use_cuda:
            # wrap input tensors in cuda tensors
            sample = utils.make_variable(sample, cuda=cuda)

        ## part I: use gradient policy method to train the generator

        # use policy gradient training when random.random() > 50%
        if random.random()  >= 0.5:

            print("Policy Gradient Training")
            
            sys_out_batch = generator(sample) # 64 X 50 X 6632

            out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 * 50) X 6632   
             
            _,prediction = out_batch.topk(1)
            prediction = prediction.squeeze(1) # 64*50 = 3200
            prediction = torch.reshape(prediction, sample['net_input']['src_tokens'].shape) # 64 X 50
            
            with torch.no_grad():
                reward = discriminator(sample['net_input']['src_tokens'], prediction) # 64 X 1

            train_trg_batch = sample['target'] # 64 x 50
            
            pg_loss = pg_criterion(sys_out_batch, train_trg_batch, reward, use_cuda)
            sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens'] # 64
            logging_loss = pg_loss / math.log(2)
            g_logging_meters['train_loss'].update(logging_loss.item(), sample_size)
            logging.debug(f"G policy gradient loss at batch {i}: {pg_loss.item():.3f}, lr={g_optimizer.param_groups[0]['lr']}")
            g_optimizer.zero_grad()
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
            g_optimizer.step()

        else:
            # MLE training
            print("MLE Training")

            sys_out_batch = generator(sample)

            out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  

            train_trg_batch = sample['target'].view(-1) # 64*50 = 3200

            loss = g_criterion(out_batch, train_trg_batch)

            sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
            nsentences = sample['target'].size(0)
            logging_loss = loss.data / sample_size / math.log(2)
            g_logging_meters['bsz'].update(nsentences)
            g_logging_meters['train_loss'].update(logging_loss, sample_size)
            logging.debug(f"G MLE loss at batch {i}: {g_logging_meters['train_loss'].avg:.3f}, lr={g_optimizer.param_groups[0]['lr']}")
            g_optimizer.zero_grad()
            loss.backward()
            # all-reduce grads and rescale by grad_denom
            for p in generator.parameters():
                if p.requires_grad:
                    p.grad.data.div_(sample_size)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
            g_optimizer.step()

        num_update += 1


        # part II: train the discriminator
        bsz = sample['target'].size(0) # batch_size = 64
    
        src_sentence = sample['net_input']['src_tokens'] # 64 x max-len i.e 64 X 50

        # now train with machine translation output i.e generator output
        true_sentence = sample['target'].view(-1) # 64*50 = 3200
        
        true_labels = Variable(torch.ones(sample['target'].size(0)).float()) # 64 length vector

        with torch.no_grad():
            sys_out_batch = generator(sample) # 64 X 50 X 6632

        out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
            
        _,prediction = out_batch.topk(1)
        prediction = prediction.squeeze(1)  #64 * 50 = 6632
        
        fake_labels = Variable(torch.zeros(sample['target'].size(0)).float()) # 64 length vector

        fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

        if use_cuda:
            fake_labels = fake_labels.cuda()
        
        disc_out = discriminator(src_sentence, fake_sentence) # 64 X 1
        
        d_loss = d_criterion(disc_out.squeeze(1), fake_labels)

        acc = torch.sum(torch.round(disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)

        d_logging_meters['train_acc'].update(acc)
        d_logging_meters['train_loss'].update(d_loss)
        logging.debug(f"D training loss {d_logging_meters['train_loss'].avg:.3f}, acc {d_logging_meters['train_acc'].avg:.3f} at batch {i}")
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()



    # validation
    # set validation mode
    generator.eval()
    discriminator.eval()
    # Initialize dataloader
    max_positions_valid = (args.fixed_max_len, args.fixed_max_len)
    valloader = dataset.eval_dataloader(
        'valid',
        max_tokens=args.max_tokens,
        max_sentences=args.joint_batch_size,
        max_positions=max_positions_valid,
        skip_invalid_size_inputs_valid_test=True,
        descending=True,  # largest batch first to warm the caching allocator
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )

    # reset meters
    for key, val in g_logging_meters.items():
        if val is not None:
            val.reset()
    for key, val in d_logging_meters.items():
        if val is not None:
            val.reset()

    for i, sample in enumerate(valloader):

        with torch.no_grad():
            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            # generator validation
            sys_out_batch = generator(sample)
            out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
            dev_trg_batch = sample['target'].view(-1) # 64*50 = 3200

            loss = g_criterion(out_batch, dev_trg_batch)
            sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
            loss = loss / sample_size / math.log(2)
            g_logging_meters['valid_loss'].update(loss, sample_size)
            logging.debug(f"G dev loss at batch {i}: {g_logging_meters['valid_loss'].avg:.3f}")

            # discriminator validation
            bsz = sample['target'].size(0)
            src_sentence = sample['net_input']['src_tokens']
            # train with half human-translation and half machine translation

            true_sentence = sample['target']
            true_labels = Variable(torch.ones(sample['target'].size(0)).float())

            with torch.no_grad():
                sys_out_batch = generator(sample)

            out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  

            _,prediction = out_batch.topk(1)
            prediction = prediction.squeeze(1)  #64 * 50 = 6632

            fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())

            fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

            if use_cuda:
                fake_labels = fake_labels.cuda()

            disc_out = discriminator(src_sentence, fake_sentence)
            d_loss = d_criterion(disc_out.squeeze(1), fake_labels)
            acc = torch.sum(torch.round(disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)
            d_logging_meters['valid_acc'].update(acc)
            d_logging_meters['valid_loss'].update(d_loss)
            logging.debug(f"D dev loss {d_logging_meters['valid_loss'].avg:.3f}, acc {d_logging_meters['valid_acc'].avg:.3f} at batch {i}")

    torch.save(generator,
               open(checkpoints_path + f"joint_{g_logging_meters['valid_loss'].avg:.3f}.epoch_{epoch_i}.pt",
                    'wb'), pickle_module=dill)

    if g_logging_meters['valid_loss'].avg < best_dev_loss:
        best_dev_loss = g_logging_meters['valid_loss'].avg
        torch.save(generator, open(checkpoints_path + "best_gmodel.pt", 'wb'), pickle_module=dill)
