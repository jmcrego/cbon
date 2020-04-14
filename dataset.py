# -*- coding: utf-8 -*-
import logging
import yaml
import sys
import os
import io
import math
import glob
#import gzip
import random
import itertools
import pyonmttok
import numpy as np
from collections import defaultdict, Counter
from utils import open_read

class Dataset():

    def __init__(self, args, token, vocab, skip_subsampling=False):
        self.vocab = vocab
        self.batch_size = args.batch_size
        self.window = args.window
        self.n_negs = args.n_negs
        self.mode = args.mode
        self.vocab_size = len(vocab)
        self.voc_maxn = vocab.max_ngram
        self.idx_pad = vocab.idx_pad
        self.idx_unk = vocab.idx_unk
        self.corpus = []
        self.wrd2n = defaultdict(int)
        ntokens = 0
        nOOV = 0
        for file in args.data:
            f, is_gzip = open_read(file)
            for l in f:
                if is_gzip:
                    l = l.decode('utf8')
                toks = token.tokenize(l.strip(' \n'))
                idxs = []
                for tok in toks:
                    idx = vocab[tok]
                    if idx == vocab.idx_unk:
                        nOOV += 1
                    idxs.append(idx)
                    self.wrd2n[idx] += 1
                self.corpus.append(idxs)
                ntokens += len(idxs)
            f.close()
        pOOV = 100.0 * nOOV / ntokens
        logging.info('read {} sentences with {} tokens (%OOV={:.2f})'.format(len(self.corpus), ntokens, pOOV))
        ### subsample
        if not skip_subsampling:
            ntokens = self.SubSample(ntokens)
            logging.info('subsampled to {} tokens'.format(ntokens))


    def get_context(self, toks, center=None):
        ctx = []
        msk = [] #mask of positive words (to indicate true words 1.0 or padding 0.0)
        if self.window > 0 and center is not None:
            first_idx = max(center-self.window, 0)
            last_idx = min(center+self.window, len(toks)-1)
        else:
            first_idx = 0
            last_idx = len(toks)-1

        ### add all ngrams in [first_idx, last_idx] which do not contain center
        for i in range(first_idx, last_idx+1):
            if center is not None and i == center:
                continue
            ctx.append(toks[i]) #toks[i] = 34
            msk.append(True)
            ngrams = [str(toks[i])] #ngrams ['34']

            for n in range(1,self.voc_maxn):
                if center is not None and i+n == center:
                    break
                if i+n >= len(toks):
                    break
                if toks[i+n] == self.idx_unk:
                    break

                ngrams.append(str(toks[i+n])) #ngrams ['34', '28']
                ngram = self.vocab[' '.join(ngrams)] #ngram = 124
                if ngram == self.idx_unk:
                    break

                ctx.append(ngram)
                msk.append(True)

        return ctx, msk

    def get_negatives(self, wrd, ctx):
        neg = []
        while len(neg) < self.n_negs:
            idx = random.randint(2, self.vocab_size-2) #do not consider idx=0 (pad) nor idx=1 (unk)
            if idx != wrd and idx not in ctx:
                neg.append(idx)
        return neg

    def add_pad(self, batch_ctx, batch_msk):
        max_len = max([len(x) for x in batch_ctx])
        print("max_len={}".format(max_len))
        for k in range(len(batch_ctx)):
            addn = max_len - len(batch_ctx[k])
            batch_ctx[k] += [self.idx_pad]*addn
            batch_msk[k] += [False]*addn
            print(len(batch_ctx[k]),batch_ctx[k])
        return batch_ctx, batch_msk

    def __iter__(self):
        ######################################################
        ### sent #############################################
        ######################################################
        if self.mode == 'sentence-vectors':
            length = [len(self.corpus[i]) for i in range(len(self.corpus))]
            indexs = np.argsort(np.array(length))
            batch_snt = []
            batch_msk = []
            batch_ind = []
            for index in indexs:
                snt, msk = self.get_context(self.corpus[index]) ### returns context for the entire sentence
                batch_snt.append(snt)
                batch_msk.append(msk)
                batch_ind.append(index)
                ### batch filled
                if len(batch_snt) == self.batch_size:
                    batch_snt, batch_msk = self.add_pad(batch_snt, batch_msk)
                    yield [batch_snt, batch_msk, batch_ind]
                    batch_snt = []
                    batch_msk = []
                    batch_ind = []
            if len(batch_snt):
                batch_snt, batch_msk = self.add_pad(batch_snt, batch_msk)
                yield [batch_snt, batch_msk, batch_ind]

        ######################################################
        ### train ############################################
        ######################################################
        elif self.mode == 'train':
            if self.window == 0:
                length = [len(self.corpus[i]) for i in range(len(self.corpus))]
                indexs = np.argsort(np.array(length)) ### from smaller to larger sentences
            else:
                indexs = [i for i in range(len(self.corpus))]
                random.shuffle(indexs) 
            batch_wrd = []
            batch_ctx = []
            batch_neg = []
            batch_msk = []
            for index in indexs:
                toks = self.corpus[index]
                if len(toks) < 2: ### may be subsampled
                    continue
                for i in range(len(toks)):
                    wrd = toks[i]
                    ctx, msk = self.get_context(toks,i)
                    neg = self.get_negatives(wrd,ctx)
                    batch_wrd.append(wrd)
                    batch_ctx.append(ctx)
                    batch_neg.append(neg)
                    batch_msk.append(msk)
                    if len(batch_wrd) == self.batch_size:
                        batch_ctx, batch_msk = self.add_pad(batch_ctx, batch_msk)
                        yield [batch_wrd, batch_ctx, batch_neg, batch_msk]
                        batch_wrd = []
                        batch_ctx = []
                        batch_neg = []
                        batch_msk = []
            if len(batch_wrd):
                batch_ctx, batch_msk = self.add_pad(batch_ctx, batch_msk)
                yield [batch_wrd, batch_ctx, batch_neg, batch_msk]

        ######################################################
        ### error ############################################
        ######################################################
        else:
            logging.error('bad -mode option {}'.format(self.mode))
            sys.exit()


    def SubSample(self, sum_counts):
#        wrd2n = dict(Counter(list(itertools.chain.from_iterable(self.corpus))))
        wrd2p_keep = {}
        for wrd in self.wrd2n:
            p_wrd = float(self.wrd2n[wrd]) / sum_counts ### proportion of the word
            p_keep = 1e-3 / p_wrd * (1 + math.sqrt(p_wrd * 1e3)) ### probability to keep the word
            wrd2p_keep[wrd] = p_keep

        filtered_corpus = []
        ntokens = 0
        for toks in self.corpus:
            filtered_corpus.append([])
            for wrd in toks:
                if random.random() < wrd2p_keep[wrd]:
                    filtered_corpus[-1].append(wrd)
                    ntokens += 1

        self.corpus = filtered_corpus
        return ntokens

    def NegativeSamples(self):
#        wrd2n = dict(Counter(list(itertools.chain.from_iterable(self.corpus))))
        normalizing_factor = sum([v**0.75 for v in self.wrd2n.values()])
        sample_probability = {}
        for wrd in self.wrd2n:
            sample_probability[wrd] = self.wrd2n[wrd]**0.75 / normalizing_factor
        words = np.array(list(sample_probability.keys()))
        probs = np.array(list(sample_probability.values()))
        while True:
            wrd_list = []
            sampled_index = np.random.multinomial(self.n_negs, probs)
            for index, count in enumerate(sampled_index):
                for _ in range(count):
                     wrd_list.append(words[index])
            yield wrd_list




