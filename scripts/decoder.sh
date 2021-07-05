#!/bin/bash

SCRIPTS=dependencies/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

BPEROOT=dependencies/subword-nmt/subword_nmt
BPE_TOKENS=40000

l=en

DATASET=amazon-3m
BPE_CODE=data/data-fairseq/$DATASET/code

mkdir -p data/data-fairseq/$DATASET
mkdir -p data/data-fairseq/$DATASET/tmp

tmp=data/data-fairseq/$DATASET/tmp

echo "~~Invoking Mosesdecoder on train data..."
rm $tmp/train.src
rm $tmp/train.tgt

cat data/data-raw/$DATASET/trn.src | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.src

cat data/data-raw/$DATASET/trn.tgt | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tgt

echo "~~Invoking Mosesdecoder on test data..."
rm $tmp/test.src
rm $tmp/test.tgt

cat data/data-raw/$DATASET/tst.src | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> $tmp/test.src

cat data/data-raw/$DATASET/tst.tgt | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> $tmp/test.tgt

echo "~~Preparing valid data..."
rm $tmp/valid.*

cp $tmp/test.src $tmp/valid.src
cp $tmp/test.tgt $tmp/valid.tgt

echo "~~Learning BPE tokenizer on train data..."
TRAIN=$tmp/train.src-tgt
rm -f $TRAIN
cat $tmp/train.src >> $TRAIN
cat $tmp/train.tgt >> $TRAIN

echo "~~learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

echo "~~Applying BPE on train / test / valid data..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/train.src > $tmp/bpe.train.src
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/train.tgt > $tmp/bpe.train.tgt

python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.src > $tmp/bpe.test.src
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.tgt > $tmp/bpe.test.tgt

python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/valid.src > $tmp/bpe.valid.src
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/valid.tgt > $tmp/bpe.valid.tgt

echo "~~Copying data..."
mkdir -p data/data-fairseq/$DATASET/$DATASET.pre-bin
PRE_BIN=data/data-fairseq/$DATASET/$DATASET.pre-bin

cp $tmp/bpe.train.src $PRE_BIN/train.src
cp $tmp/bpe.train.tgt $PRE_BIN/train.tgt

cp $tmp/bpe.valid.src $PRE_BIN/valid.src
cp $tmp/bpe.valid.tgt $PRE_BIN/valid.tgt

cp $tmp/bpe.test.src $PRE_BIN/test.src
cp $tmp/bpe.test.tgt $PRE_BIN/test.tgt

cp data/data-fairseq/$DATASET/code $PRE_BIN/code

mkdir -p data/data-bin
mkdir -p data/data-bin/$DATASET.tokenized.src-tgt

echo "~~Invoke fairseq preprocessing"
TEXT=$PRE_BIN
fairseq-preprocess --source-lang src  --target-lang tgt \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir data/data-bin/$DATASET.tokenized.src-tgt.2
