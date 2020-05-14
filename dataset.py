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
        self.shard_size = args.shard_size
        self.vocab_size = len(vocab)
        self.voc_maxn = vocab.max_ngram
        self.idx_pad = vocab.idx_pad
        self.idx_unk = vocab.idx_unk
        self.corpus = []
        self.wrd2n = defaultdict(int)
        self.total_ngrams = defaultdict(int)
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
        #if not skip_subsampling:
        #    ntokens = self.SubSample(ntokens)
        #    logging.info('subsampled to {} tokens'.format(ntokens))


    def __iter__(self):
        ######################################################
        ### word-similarity word-vectors #####################
        ######################################################
        if self.mode == 'word-similarity' or self.mode == 'word-vectors':
            ### traverse sentences word by word (no shuffle no shards needed)
            batch_wrd = []
            for ind in range(len(self.corpus)):
                for wrd in self.corpus[ind]:
                    batch_wrd.append(wrd)
                    ### batch filled
                    if len(batch_wrd) == self.batch_size:
                        yield [batch_wrd]
                        batch_wrd = []
            if len(batch_wrd):
                yield [batch_wrd]

        ######################################################
        ### sentence-vectors #################################
        ######################################################
        elif self.mode == 'sentence-vectors':
            self.indexs = [i for i in range(len(self.corpus))]
            #no need to shuffle! random.shuffle(self.indexs)
            first_index = 0
            while first_index < len(self.indexs):
                indexs_shard, first_index = self.get_shard(first_index) # indexs_shard contains a subset of the indexs stored in self.indexs
                examples = self.get_examples_snt(indexs_shard)
                batchs = self.get_batchs_snt(examples)
                indexs_batchs = [i for i in range(len(batchs))]
                #no need to shuffle! random.shuffle(indexs_batchs)
                for ind in indexs_batchs:
                    yield batchs[ind]

        ######################################################
        ### train ############################################
        ######################################################
        elif self.mode == 'train':
            self.indexs = [i for i in range(len(self.corpus))]
            random.shuffle(self.indexs) #shuffle self.indexs
            first_index = 0
            while first_index < len(self.indexs):
                indexs_shard, first_index = self.get_shard(first_index) # indexs_shard contains a subset of the indexes stored in self.indexs
                examples = self.get_examples(indexs_shard)
                batchs = self.get_batchs(examples)
                indexs_batchs = [i for i in range(len(batchs))]
                random.shuffle(indexs_batchs) #shuffle indexs_batchs
                for ind in indexs_batchs:
                    yield batchs[ind]

        ######################################################
        ### error ############################################
        ######################################################
        else:
            logging.error('bad -mode option {}'.format(self.mode))
            sys.exit()


    def get_shard(self, first_index):
        next_index = min(first_index + self.shard_size, len(self.indexs))
        indexs_shard = self.indexs[first_index:next_index]
        logging.info('built shard with {} out of {} sentences [{}, {})'.format(len(indexs_shard), len(self.indexs), first_index, next_index))
        return indexs_shard, next_index


    def get_examples(self, indexs_shard):
        examples = [] ### ind (position in corpus), wrd (word to predict or empty), neg (n negative words or empty), ctx (context or sentence)
        for ind in indexs_shard:
            for center in range(len(self.corpus[ind])):
                wrd = self.corpus[ind][center] #idx
                if wrd < 2: ### i dont want to predict <unk> or <pad>
                    continue
                ctx, neg = self.get_ctx_neg(self.corpus[ind], center) #[idx, idx, ...], [idx, idx, ...]
                if len(ctx)==0:
                    continue
                e = []
                #logging.info('wrd: {}'.format(wrd))
                #logging.info('ctx: {}'.format(ctx))
                #logging.info('neg: {}'.format(neg))
                e.append(wrd) #the word to predict
                e.extend(neg) #n_negs negative words
                e.extend(ctx) #ngrams around [center-window, center+window] used to predict
                examples.append(e)
        logging.info('compiled shard with {} examples'.format(len(examples)))
        for n,N in self.total_ngrams.items():
            logging.info('ctx {}-grams: {}'.format(n,N))
        return examples

    def get_examples_snt(self, indexs_shard):
        examples = [] ### ind (position in corpus), wrd (word to predict or empty), neg (n negative words or empty), ctx (context or sentence)
        for ind in indexs_shard:
            snt = self.get_snt(self.corpus[ind]) #[idx, idx, ...], [i]
#            if len(snt)==0: # at least one ngram in snt
#                continue
            e = []
            #logging.info('snt: {}'.format(snt))
            e.append(ind) #position in corpus
            e.extend(snt) #ngrams in sentence
            examples.append(e)
        logging.info('compiled shard with {} examples'.format(len(examples)))
        for n,N in self.total_ngrams.items():
            logging.info('ctx {}-grams: {}'.format(n,N))
        return examples


    def get_batchs(self, examples):
        ### sort examples by len
        logging.info('sorting examples in shard by number of ngrams to minimize padding')
        length = [len(examples[k]) for k in range(len(examples))] #length of sentences in this shard
        index_examples = np.argsort(np.array(length)) ### These are indexs of examples

        batchs = []
        batch_wrd = []
        batch_ctx = []
        batch_neg = []
        batch_msk = []
        for ind in index_examples:
            e = examples[ind]
            wrd = e[0]
            neg = e[1:self.n_negs+1]
            ctx = e[self.n_negs+1:]
            #if len(ctx) == 0:
            #    logging.error('ctx length is 0')
            #    sys.exit()
            msk = [True] * len(ctx)
            batch_wrd.append(wrd)
            batch_ctx.append(ctx)
            batch_neg.append(neg)
            batch_msk.append(msk)
            if len(batch_wrd) == self.batch_size:
                batch_ctx, batch_msk = self.add_pad(batch_ctx, batch_msk)
                batchs.append([batch_wrd, batch_ctx, batch_neg, batch_msk])
                batch_wrd = []
                batch_ctx = []
                batch_neg = []
                batch_msk = []
        if len(batch_wrd):
            batch_ctx, batch_msk = self.add_pad(batch_ctx, batch_msk)
            batchs.append([batch_wrd, batch_ctx, batch_neg, batch_msk])
        logging.info('found {} batchs in shard'.format(len(batchs)))
        return batchs


    def get_batchs_snt(self, examples):
        ### this shard is built of indexs_shard, indexes that points to self.indexs
        logging.info('sorting examples in shard by number of ngrams to minimize padding')
        length = [len(examples[k]) for k in range(len(examples))] #length of sentences in this shard
        index_examples = np.argsort(np.array(length)) ### These are indexs of examples

        batchs = []
        batch_snt = []
        batch_msk = []
        batch_ind = []
        for ind in index_examples:
            e = examples[ind]
            ind = e[0]
            snt = e[1:]
            msk = [True] * len(snt)
            batch_ind.append(ind) ### position in original corpus
            batch_snt.append(snt)
            batch_msk.append(msk)
            ### batch filled
            if len(batch_snt) == self.batch_size:
                batch_snt, batch_msk = self.add_pad(batch_snt, batch_msk)
                batchs.append([batch_snt, batch_msk, batch_ind])
                batch_snt = []
                batch_msk = []
                batch_ind = []
        if len(batch_snt):
            batch_snt, batch_msk = self.add_pad(batch_snt, batch_msk)
            batchs.append([batch_snt, batch_msk, batch_ind])
        return batchs

    def get_ctx_neg(self, idxs, center):
        toks = [self.vocab[idxs[i]] for i in range(len(idxs))]
        #logging.info('center={}:{} {}'.format(center, toks[center], ' '.join(toks)))

        if self.window > 0:
            beg = max(center - self.window, 0)
            end = min(center + self.window + 1, len(idxs))
        else:
            beg = 0
            end = len(idxs)

        ###
        ### find all ngrams
        ###
        ctx = []
        for i in range(beg, end): 
            if i == center:
                continue
            if idxs[i] == self.idx_unk:
                continue

            self.total_ngrams[1] += 1

            ctx.append(idxs[i])
            #logging.info('[{}] 1-gram_idx={} \'{}\''.format(i, idxs[i], toks[i]))

            for j in range(i+2,i+self.voc_maxn+1): 
                if j > len(idxs):
                    break
                if center>=i and center<j:
                    break
                if self.idx_unk in idxs[i:j]:
                    break

                ngram = ' '.join([str(k) for k in idxs[i:j]])
                idx = self.vocab[ngram]
                if idx != self.idx_unk:
                    ctx.append(idx)
                    self.total_ngrams[j-i] += 1

                    #logging.info('[{}:{}) {}-gram_idx={} \'{}\''.format(i, j, j-i, idx, ngram))

        ###
        ### find Negative words
        ###
        neg = []
        while len(neg) < self.n_negs:
            idx = random.randint(2, self.vocab_size-2) #do not consider idx=0 (pad) nor idx=1 (unk)
            if idx != idxs[center] and idx not in ctx:
                neg.append(idx)

        return ctx, neg

    def get_snt(self, idxs):
        toks = [self.vocab[idxs[i]] for i in range(len(idxs))]
        #logging.info('center={}:{} {}'.format(center, toks[center], ' '.join(toks)))
        beg = 0
        end = len(idxs)
        ###
        ### find all ngrams
        ###
        snt = []
        for i in range(beg, end): 
            if idxs[i] == self.idx_unk:
                continue
            self.total_ngrams[1] += 1
            snt.append(idxs[i])
            #logging.info('[{}] 1-gram_idx={} \'{}\''.format(i, idxs[i], toks[i]))
            for j in range(i+2,i+self.voc_maxn+1): 
                if j > len(idxs):
                    break
                if self.idx_unk in idxs[i:j]:
                    break
                ngram = ' '.join([str(k) for k in idxs[i:j]])
                idx = self.vocab[ngram]
                if idx != self.idx_unk:
                    snt.append(idx)
                    self.total_ngrams[j-i] += 1
                    #logging.info('[{}:{}) {}-gram_idx={} \'{}\''.format(i, j, j-i, idx, ngram))
        return snt


    def add_pad(self, batch_ctx, batch_msk):
        max_len = max([len(x) for x in batch_ctx])
        #logging.info('max_len={} lens: {}'.format(max_len, [len(x) for x in batch_ctx]))
        for k in range(len(batch_ctx)):
            addn = max_len - len(batch_ctx[k])
            batch_ctx[k] += [self.idx_pad]*addn
            batch_msk[k] += [False]*addn
            #print(len(batch_ctx[k]),batch_ctx[k])
        return batch_ctx, batch_msk


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




