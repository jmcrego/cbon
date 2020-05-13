# -*- coding: utf-8 -*-
import torch
import logging
import yaml
import sys
import os
import io
import math
import random
import itertools
import pyonmttok
import glob
import numpy as np
import torch.nn as nn
from collections import Counter
from dataset import Dataset
from vocab import Vocab
from tokenizer import OpenNMTTokenizer

min_sigmoid = 1e-06
max_sigmoid = 1.0 - 1e-06

def save_model(pattern, model, n_steps, keep_last_n):
    file = pattern + '.model.{:09d}.pth'.format(n_steps)
    state = {
        'pooling': model.pooling,
        'embedding_size': model.ds,
        'vocab_size': model.vs,
        'idx_pad': model.idx_pad,
        'n_steps': n_steps,
        'model': model.state_dict()
    }
    torch.save(state, file)
    logging.info('saved model checkpoint {}'.format(file))
    files = sorted(glob.glob(pattern + '.model.?????????.pth')) 
    while len(files) > keep_last_n:
        f = files.pop(0)
        os.remove(f) ### first is the oldest
        logging.debug('removed checkpoint {}'.format(f))

def load_model(pattern, vocab):
    model = None
    n_steps = 0
    files = sorted(glob.glob(pattern + '.model.?????????.pth')) 
    if len(files):
        file = files[-1] ### last is the newest
        checkpoint = torch.load(file)
        pooling = checkpoint['pooling']
        embedding_size = checkpoint['embedding_size']
        n_steps = checkpoint['n_steps']
        vocab_size = checkpoint['vocab_size']
        if vocab_size != len(vocab):
            logging.error('incompatible vocabulary size {} != {}'.format(vocab_size, len(vocab)))
            sys.exit()
        idx_pad = checkpoint['idx_pad']
        if idx_pad != vocab.idx_pad:
            logging.error('incompatible idx_pad {} != {}'.format(idx_pad, vocab.idx_pad))
            sys.exit()
        model = Word2Vec(vocab_size, embedding_size, pooling, idx_pad)
        model.load_state_dict(checkpoint['model'])
        logging.info('loaded checkpoint {} [{},{}] pooling={}'.format(file,len(vocab),embedding_size,pooling))
    return model, n_steps

def save_optim(pattern, optimizer):
    file = pattern + '.optim.pth'
    state = {
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, file)
    logging.info('saved optim in {}'.format(file))

def load_build_optim(pattern, model, lr, beta1, beta2, eps):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=0.01, amsgrad=False)
    file = pattern + '.optim.pth'
    if os.path.exists(file): 
        checkpoint = torch.load(file)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info('loaded optimizer from {}.optim.pth'.format(pattern))
    else:
        logging.info('build optimizer from scratch')
    return optimizer

def sequence_mask(lengths):
    lengths = np.array(lengths)
    bs = len(lengths)
    l = lengths.max()
    msk = np.cumsum(np.ones([bs,l],dtype=int), axis=1).T #[l,bs] (transpose to allow combine with lenghts)
    mask = (msk <= lengths) ### i use lenghts-1 because the last unpadded word is <eos> and i want it masked too
    return mask.T #[bs,l]

####################################################################
### Word2Vec #######################################################
####################################################################
class Word2Vec(nn.Module):
    def __init__(self, vs, ds, pooling, idx_pad):
        super(Word2Vec, self).__init__()
        self.vs = vs
        self.ds = ds
        self.pooling = pooling
        self.idx_pad = idx_pad
        self.iEmb = nn.Embedding(self.vs, self.ds, padding_idx=self.idx_pad)
        self.oEmb = nn.Embedding(self.vs, self.ds, padding_idx=self.idx_pad)
        #nn.init.xavier_uniform_(self.iEmb.weight)
        #nn.init.xavier_uniform_(self.oEmb.weight)
        nn.init.uniform_(self.iEmb.weight, -0.1, 0.1)
        nn.init.uniform_(self.oEmb.weight, -0.1, 0.1)

    def WordEmbed(self, wrd, layer):
        wrd = torch.as_tensor(wrd)
        if self.iEmb.weight.is_cuda:
            wrd = wrd.cuda()

        if layer == 'iEmb':
            emb = self.iEmb(wrd) #[bs,ds]
        elif layer == 'oEmb':
            emb = self.oEmb(wrd) #[bs,ds]
        else:
            logging.error('bad layer {}'.format(layer))
            sys.exit()
 
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            logging.error('NaN/Inf detected in {} layer emb.shape={}\nwrds {}'.format(layer,emb.shape,wrd))
            self.NaN(wrd,emb)
            sys.exit()
        return emb

    def NgramsEmbed(self, ngrams, msk):
        ngrams_emb = self.WordEmbed(ngrams,'iEmb') #[bs,n,ds]
        if self.pooling == 'avg':
            ngrams_emb = (ngrams_emb*msk.unsqueeze(-1)).sum(1) / torch.sum(msk, dim=1).unsqueeze(-1) #[bs,n,ds]x[bs,n,1]=>[bs,ds] / [bs,1] = [bs,ds] 
        elif self.pooling == 'sum':
            ngrams_emb = (ngrams_emb*msk.unsqueeze(-1)).sum(1) #[bs,n,ds]x[bs,n,1]=>[bs,ds]
        elif self.pooling == 'max':
            ngrams_emb, _ = torch.max(ngrams_emb*msk + (1.0-msk)*-999.9, dim=1) #-999.9 should be -Inf but it produces a nan when multiplied by 0.0            
        else:
            logging.error('bad -pooling option {}'.format(self.pooling))
            sys.exit()
        return ngrams_emb

    def forward(self, batch):
        #batch[0] : batch of center words (list:bs)
        #batch[1] : batch of context words (list:bs of list:nc)
        #batch[2] : batch of negative words (list:bs of list:nn)
        #batch[3] : batch of masks for context words (list:bs of list:nc)
        msk = torch.as_tensor(batch[3]) #[bs,n] (positive words are 1.0 others are 0.0)
        if self.iEmb.weight.is_cuda:
            msk = msk.cuda()
        ###
        #Context words are embedded using iEmb
        ###
        ctx_emb = self.NgramsEmbed(batch[1], msk) #[bs,ds]
        ###
        #Center words are embedded using oEmb
        ###
        wrd_emb = self.WordEmbed(batch[0],'oEmb') #[bs,ds]
        ###
        #Negative words are embedded using oEmb
        ###
        neg_emb = self.WordEmbed(batch[2],'oEmb').neg() #[bs,nn,ds]
        ###
        ### computing positive words loss
        ###
        #i use clamp to prevent NaN/Inf appear when computing the log of 1.0/0.0
        err = torch.bmm(ctx_emb.unsqueeze(1), wrd_emb.unsqueeze(-1)).squeeze().sigmoid().clamp(min_sigmoid, max_sigmoid).log().neg() #[bs,1,ds] x [bs,ds,1] = [bs,1] = > [bs]
        loss = err.mean() # mean errors of examples in this batch
        ###
        ### computing negative words loss
        ###
        err = torch.bmm(ctx_emb.unsqueeze(1), neg_emb.transpose(2,1)).squeeze().sigmoid().clamp(min_sigmoid, max_sigmoid).log().neg() #[bs,1,ds] x [bs,ds,n] = [bs,1,n] = > [bs,n]
        err = torch.sum(err, dim=1) #[bs] (sum of errors of all negative words) (not averaged)
        loss += err.mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logging.error('NaN/Inf detected in cbow_loss for batch {}'.format(batch))
            sys.exit()
        return loss

