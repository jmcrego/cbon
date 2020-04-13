#!/bin/bash

download(){
    mkdir -p $DIR
    wget http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-fr.tsv.gz
    mv news-commentary-v14.en-fr.tsv.gz $DIR/
    gunzip $DIR/news-commentary-v14.en-fr.tsv.gz
    cut -f 1 $DIR/news-commentary-v14.en-fr.tsv > $DIR/news-commentary-v14.en
    cut -f 2 $DIR/news-commentary-v14.en-fr.tsv > $DIR/news-commentary-v14.fr
    rm -f $DIR/news-commentary-v14.en-fr.tsv
    echo -e "mode: conservative\njoiner_annotate: true" > $DIR/token_options
}

DIR=$PWD/raw

#download
#python3 cbon.py -name ex1 -mode preprocess -data $DIR/news-commentary-v14.en -voc_maxn 3 -voc_minf 10 -tok_conf $DIR/token_options -log_file stderr
#python3 cbon.py -name ex1 -mode train -data $DIR/news-commentary-v14.en -log_file stderr -skip_subsampling -embedding_size 100 -batch_size 8

