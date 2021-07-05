#!/bin/bash

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

l=en

mkdir amazoncat-13k
mkdir amazoncat-13k/tmp

tmp=amazoncat-13k/tmp

echo "invoking mosesdecoder on train data..."
rm amazoncat-13k/tmp/train.tags.src-tgt.tok.src
cat data/trn.src | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> amazoncat-13k/tmp/train.tags.src-tgt.tok.src

rm amazoncat-13k/tmp/train.tags.src-tgt.tok.tgt
cat data/trn.tgt | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> amazoncat-13k/tmp/train.tags.src-tgt.tok.tgt

echo "invoking mosesdecoder on test data..."
rm amazoncat-13k/tmp/test.src
cat data/tst.src | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> amazoncat-13k/tmp/test.src

rm amazoncat-13k/tmp/test.tgt
cat data/tst.tgt | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> amazoncat-13k/tmp/test.tgt

echo "splitting train data into train and valid..."
awk '{if (NR%10 == 0)  print $0; }' $tmp/train.tags.src-tgt.tok.src > $tmp/valid.src
awk '{if (NR%10 != 0)  print $0; }' $tmp/train.tags.src-tgt.tok.src > $tmp/train.src

awk '{if (NR%10 == 0)  print $0; }' $tmp/train.tags.src-tgt.tok.tgt > $tmp/valid.tgt
awk '{if (NR%10 != 0)  print $0; }' $tmp/train.tags.src-tgt.tok.tgt > $tmp/train.tgt

BPE_CODE=amazoncat-13k/code

TRAIN=$tmp/train.src-tgt
rm -f $TRAIN
cat $tmp/train.src >> $TRAIN
cat $tmp/train.tgt >> $TRAIN

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

echo "applying BPE on train / test / valid data..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/train.src > $tmp/bpe.train.src
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/train.tgt > $tmp/bpe.train.tgt

python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.src > $tmp/bpe.test.src
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.tgt > $tmp/bpe.test.tgt

python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/valid.src > $tmp/bpe.valid.src
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/valid.tgt > $tmp/bpe.valid.tgt

# perl $CLEAN -ratio 1.5 $tmp/bpe.train src tgt amazoncat-13k/train 1 10
# perl $CLEAN -ratio 1.5 $tmp/bpe.valid src tgt amazoncat-13k/valid 1 10

cp $tmp/bpe.train.src amazoncat-13k/train.src
cp $tmp/bpe.train.tgt amazoncat-13k/train.tgt

cp $tmp/bpe.valid.src amazoncat-13k/valid.src
cp $tmp/bpe.valid.tgt amazoncat-13k/valid.tgt

cp $tmp/bpe.test.src amazoncat-13k/test.src
cp $tmp/bpe.test.tgt amazoncat-13k/test.tgt

conda activate fairseq

mkdir data-bin
mkdir data-bin/amazoncat-13k.tokenized.src-tgt

echo "invoke fairseq preprocessing"
TEXT=./amazoncat-13k
fairseq-preprocess 
    --source-lang src 
    --target-lang tgt 
    --trainpref $TEXT/train 
    --validpref $TEXT/valid 
    --testpref $TEXT/test 
    --destdir data-bin/amazoncat-13k.tokenized.src-tgt