# -*- coding: utf-8 -*-
import logging
import yaml
import sys
import os
#import io
#import math
#import glob
#import gzip
#import random
#import itertools
import pyonmttok
#import numpy as np
#from collections import defaultdict, Counter

class OpenNMTTokenizer():

    def __init__(self, fyaml):
        opts = {}
        if fyaml is None:
            self.tokenizer = None
        else:
            if not os.path.exists(fyaml):
                logging.error('missing {} file'.format(fyaml))
                sys.exit()
            with open(fyaml) as yamlfile: 
                opts = yaml.load(yamlfile, Loader=yaml.FullLoader)

            if 'mode' not in opts:
                logging.error('error: missing mode in tokenizer')
                sys.exit()

            mode = opts["mode"]
            del opts["mode"]
            self.tokenizer = pyonmttok.Tokenizer(mode, **opts)
            logging.info('built tokenizer mode={} {}'.format(mode,opts))

    def tokenize(self, text):
        if self.tokenizer is None:
            tokens = text.split()
        else:
            tokens, _ = self.tokenizer.tokenize(text)
        return tokens

    def detokenize(self, tokens):
        if self.tokenizer is None:
            return tokens
        return self.tokenizer.detokenize(tokens)



