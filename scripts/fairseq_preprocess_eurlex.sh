TEXT=data/data-raw/eurlex-4k/preprocessed
rm -rf data/data-bin/$DATASET.tokenized.src-tgt
fairseq-preprocess --source-lang src  --target-lang tgt \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir data/data-bin/$DATASET.tokenized.src-tgt
