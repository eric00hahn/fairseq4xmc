# arch option : transformer, fconv_iwslt_de_en

MODEL=fconv_self_att
DATASET=eurlex-4k

mkdir -p checkpoints/$DATASET.$MODEL.2

CUDA_VISIBLE_DEVICES=0 fairseq-train data/data-bin/$DATASET.tokenized.src-tgt.2 \
    --lr 0.1 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch $MODEL --save-dir checkpoints/$DATASET.$MODEL.2 --optimizer nag \
    --skip-invalid-size-inputs-valid-test
