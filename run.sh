#!/bin/bash

download(){
    mkdir -p $DIR
	rsync -avzP ssaling02:~/raw/Europarl/enfr/Europarl.en-fr.en $DIR/
	rsync -avzP ssaling02:~/raw/ECB/enfr/ECB.en-fr.en $DIR/
	rsync -avzP ssaling02:~/raw/EMEA/enfr/EMEA.en-fr.en $DIR/
	rsync -avzP ssaling02:~/raw/news-commentary/enfr/news-commentary-v14.en $DIR/
	rsync -avzP ssaling02:~/raw/JRC-Acquis/enfr/JRC-Acquis.en-fr.en $DIR/
	rsync -avzP ssaling02:~/raw/TED2013/enfr/TED2013.en-fr.en $DIR/
	rsync -avzP ssaling02:~/raw/Wikipedia/enfr/Wikipedia.en-fr.en $DIR/
    echo -e "mode: conservative\njoiner_annotate: true" > $DIR/token_options
}

DIR=$PWD/raw

#
download
#python3 cbon.py -name ex1 -mode preprocess -data $DIR/news-commentary-v14.en -voc_maxn 3 -voc_minf 10 -tok_conf $DIR/token_options -log_file stderr
#python3 cbon.py -name ex1 -mode train -data $DIR/news-commentary-v14.en -log_file stderr -skip_subsampling -embedding_size 100 -batch_size 8

