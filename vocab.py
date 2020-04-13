# -*- coding: utf-8 -*-
import logging
#import yaml
import sys
import os
import io
#import math
#import glob
#import gzip
#import random
#import itertools
#import pyonmttok
#import numpy as np
from collections import defaultdict
from utils import open_read

class Vocab():

    def __init__(self):
        self.idx_unk = 0 
        self.str_unk = '<unk>'
        self.idx_pad = 0 
        self.str_pad = '<pad>'
        self.tok_to_idx = {} 
        self.idx_to_tok = [] 
        self.max_ngram = 1

    def read(self, file):
        if not os.path.exists(file):
            logging.error('missing {} file'.format(file))
            sys.exit()

        f, is_gzip = open_read(file)
        params_line = True
        for l in f:
            if is_gzip:
                l = l.decode('utf8')
            tok = l.strip(' \n')
            if params_line:
                params_line = False
                if tok.find('max_ngram=') == 0:
                    self.max_ngram = int(tok[10:])
                continue
            if tok not in self.tok_to_idx:
                self.idx_to_tok.append(tok)
                self.tok_to_idx[tok] = len(self.tok_to_idx)
        f.close()
        logging.info('read vocab ({} entries) from {}'.format(len(self.idx_to_tok),file))

    def dump(self, file):
        f = open(file, "w")
        f.write('max_ngram={}\n'.format(self.max_ngram))
        for tok in self.idx_to_tok:
            f.write(tok+'\n')
        f.close()
        logging.info('written vocab ({} entries) into {}'.format(len(self.idx_to_tok),file))

    def build(self,files,token,min_freq=5,max_size=0,max_ngram=1):
        self.max_ngram = max_ngram
        self.tok_to_frq = defaultdict(int)
        for file in files:
            f, is_gzip = open_read(file)
            for l in f:
                if is_gzip:
                    l = l.decode('utf8')
                toks = []
                for tok in token.tokenize(l.strip(' \n')):
                    toks.append(tok)
                    for n in range(max_ngram): #if max_ngram is 2 then n is 0, 1
                        if len(toks) > n:
                            ngram = ' '.join(toks[len(toks)-n-1:])
                            self.tok_to_frq[ngram] += 1

            f.close()
        self.tok_to_idx[self.str_pad] = self.idx_pad #0
        self.idx_to_tok.append(self.str_pad)        
        self.tok_to_idx[self.str_unk] = self.idx_unk #1
        self.idx_to_tok.append(self.str_unk)        
        for wrd, frq in sorted(self.tok_to_frq.items(), key=lambda item: item[1], reverse=True):
            if len(self.idx_to_tok) == max_size:
                break
            if frq < min_freq:
                break
            self.tok_to_idx[wrd] = len(self.idx_to_tok)
            self.idx_to_tok.append(wrd)
        logging.info('built vocab ({} entries) from {}'.format(len(self.idx_to_tok),files))

    def __len__(self):
        return len(self.idx_to_tok)

    def __iter__(self):
        for tok in self.idx_to_tok:
            yield tok

    def __contains__(self, s): ### implementation of the method used when invoking : entry in vocab
        if type(s) == int: ### testing an index
            return s>=0 and s<len(self)
        ### testing a string
        return s in self.tok_to_idx

    def __getitem__(self, s): ### implementation of the method used when invoking : vocab[entry]
        if type(s) == int: ### input is an index, i want the string
            if s not in self:
                logging.error("key \'{}\' not found in vocab".format(s))
                sys.exit()
            return self.idx_to_tok[s]
        ### input is a string, i want the index
        if s not in self: 
            return self.idx_unk
        return self.tok_to_idx[s]



