#!/bin/bash
python run_mlm.py \
    --model_name_or_path bert-large-uncased \
    --train_file ./data/THG_twitter/twitter.2021.train.src_after_cleaning.txt \
    --validation_file ./data/THG_twitter/twitter.2021.valid.src_after_cleaning.txt \
    --per_device_train_batch_size 96 \
    --per_device_eval_batch_size 96 \
    --do_train \
    --do_eval \
    --line_by_line \
    --output_dir ./PLM_checkpoint/bert_large_retrieval