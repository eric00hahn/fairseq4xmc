MODEL=fconv_self_att
DATASET=amazon-3m

mkdir -p checkpoints/$DATASET.$MODEL

CUDA_VISIBLE_DEVICES=0 fairseq-train data/data-bin/$DATASET.tokenized.src-tgt.2 \
    --lr 0.05 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch $MODEL --save-dir checkpoints/$DATASET.$MODEL --optimizer nag
