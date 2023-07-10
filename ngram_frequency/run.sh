#!/bin/bash
set -e
CORPUS="../data/processed/sample_processed.txt"
NGRAM="../data/processed/ngram_frequency.csv"
st="1e-7"
ep="1e-7"
make
./main --corpus_path=$CORPUS \
       --ngram_count_path=$NGRAM \
       --max_ngram_size=4 \
       --n_jobs=8 \
       --threshold=$st \
       --epsilon=$ep