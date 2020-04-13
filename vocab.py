# -*- coding: utf-8 -*-
import logging
import sys
import os
import io
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
        for l in f:
            if is_gzip:
                l = l.decode('utf8')
            tok = l.strip(' \n')
            if tok not in self.tok_to_idx:
                self.idx_to_tok.append(tok)
                self.tok_to_idx[tok] = len(self.tok_to_idx)
        f.close()
        logging.info('read vocab ({} entries) from {}'.format(len(self.idx_to_tok),file))

    def dump(self, file):
        f = open(file, "w")
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
        unigrams = 0
        bigrams = 0
        trigrams = 0
        ### build vocab
        self.tok_to_idx[self.str_pad] = self.idx_pad #0
        self.idx_to_tok.append(self.str_pad)        
        self.tok_to_idx[self.str_unk] = self.idx_unk #1
        self.idx_to_tok.append(self.str_unk)        
        for wrd, frq in sorted(self.tok_to_frq.items(), key=lambda item: item[1], reverse=True):
            if len(self.idx_to_tok) == max_size:
                break
            if frq < min_freq:
                break
            WRD = wrd.split(' ')
            if len(WRD) == 1:
                self.tok_to_idx[wrd] = len(self.idx_to_tok)
                self.idx_to_tok.append(wrd)
                unigrams += 1
            else:
                idx = [str(self.tok_to_idx[w]) for w in WRD]
                idx = ' '.join(idx)
                self.tok_to_idx[idx] = len(self.idx_to_tok)
                self.idx_to_tok.append(idx)
                if len(WRD) == 2:
                    bigrams += 1
                elif len(WRD) == 3:
                    trigrams += 1

        logging.info('built vocab ({} entries) unigrams={} bigrams={} trigrams={} from {}'.format(len(self.idx_to_tok),unigrams, bigrams, trigrams, files))

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



