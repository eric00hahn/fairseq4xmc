fairseq-generate data/data-bin/eurlex-4k.tokenized.src-tgt.2 \
    --path checkpoints/eurlex-4k.fconv_self_att.2/checkpoint_best.pt \
    --batch-size 128
    --beam 20
