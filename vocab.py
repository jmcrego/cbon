# -*- coding: utf-8 -*-
import logging
import sys
import os
import io
from collections import defaultdict
from utils import open_read

class Vocab():

    def __init__(self):
        self.idx_pad = 0 
        self.str_pad = '<pad>'
        self.idx_unk = 1 
        self.str_unk = '<unk>'
        self.tok_to_idx = {} 
        self.idx_to_tok = [] 
        self.max_ngram = 1
        self.use_bos_eos = False

    def read(self, file):
        if not os.path.exists(file):
            logging.error('missing {} file'.format(file))
            sys.exit()

        first_line = True
        f, is_gzip = open_read(file)
        for l in f:
            if is_gzip:
                l = l.decode('utf8')
            tok = l.strip(' \n')
            if first_line:
                info = tok.split()
                if len(info)!=2:
                    logging.error('erroneous first line in vocab (must contain: ngram use_bos_eos)')
                    sys.exit()
                self.max_ngram = int(info[0])
                self.use_bos_eos = bool(info[1])
                first_line = False
                continue
            if tok not in self.tok_to_idx:
                self.idx_to_tok.append(tok)
                self.tok_to_idx[tok] = len(self.tok_to_idx)
        f.close()
        logging.info('read vocab ({} entries, max_ngram={} use_bos_eos={}) from {}'.format(len(self.idx_to_tok), self.max_ngram, self.use_bos_eos, file))

    def dump(self, file):
        f = open(file, "w")
        f.write(str(self.max_ngram)+' '+str(self.use_bos_eos)+'\n')
        for tok in self.idx_to_tok:
            f.write(tok+'\n')
        f.close()
        logging.info('written vocab ({} entries, max_ngram={} use_bos_eos={}) into {}'.format(len(self.idx_to_tok), self.max_ngram, self.use_bos_eos, file))

    def build(self,files,token,min_freq=5,max_size=0,max_ngram=1,use_bos_eos=False):
        self.max_ngram = max_ngram
        self.use_bos_eos = use_bos_eos
        self.tok_to_frq = defaultdict(int)
        for file in files:
            f, is_gzip = open_read(file)
            for l in f:
                if is_gzip:
                    l = l.decode('utf8')
                toks = [] 
                sentoks = token.tokenize(l.strip(' \n'))
                if self.use_bos_eos:
                    sentoks.insert(0,'<bos>')
                    sentoks.append('<eos>')

                for tok in sentoks:
                    toks.append(tok)
                    for n in range(max_ngram): #if max_ngram is 2 then n is 0, 1
                        if len(toks) > n:
                            ngram = ' '.join(toks[len(toks)-n-1:])
                            self.tok_to_frq[ngram] += 1

            f.close()
        ngrams = defaultdict(int)
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
            ngrams[len(WRD)] += 1
            if len(WRD) == 1:
                self.tok_to_idx[wrd] = len(self.idx_to_tok)
                self.idx_to_tok.append(wrd)
            else:
                idx = [str(self.tok_to_idx[w]) for w in WRD]
                idx = ' '.join(idx)
                self.tok_to_idx[idx] = len(self.idx_to_tok)
                self.idx_to_tok.append(idx)

        logging.info('built vocab ({} entries) ngrams {} from {}'.format(len(self.idx_to_tok),dict(ngrams),files))

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
            ### s exists in self.idx_to_tok
            return self.idx_to_tok[s]
        ### input is a string, i want the index
        if s not in self: 
            return self.idx_unk
        return self.tok_to_idx[s]


    def strngram(self, s, sep=' '): ### implementation of the method used when invoking : vocab[entry]
        if type(s) != int: ### input is an index, i want the string
            logging.error("key \'{}\' is not int".format(s))
            sys.exit()

        if s not in self:
            logging.error("key \'{}\' not found in vocab".format(s))
            sys.exit()

        wrd = self.idx_to_tok[s]
        WRD = wrd.split(' ')
        chk = []
        if len(WRD) > 1:
            for w in WRD:
                chk.append(self.idx_to_tok[int(w)])
        else:
            chk.append(wrd)
        return ' '.join(chk)










