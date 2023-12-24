#!/bin/bash
python3 main.py --dataset THG \
    --exp_version lr_3e-4_bs16_epoch10_bm25retrieval_concattop5 \
    --train_src_file data/THG_twitter/twitter.2021.train.src_after_cleaning.txt \
    --train_dst_file data/THG_twitter/twitter.2021.train.dst_after_cleaning.txt \
    --val_src_file data/THG_twitter/twitter.2021.valid.src_after_cleaning.txt \
    --val_dst_file data/THG_twitter/twitter.2021.valid.dst_after_cleaning.txt \
    --test_src_file data/THG_twitter/twitter.2021.test.src_after_cleaning.txt \
    --test_dst_file data/THG_twitter/twitter.2021.test.dst_after_cleaning.txt \
    --model_name_or_path t5-base \
    --tokenizer_name_or_path t5-base \
    --do_train \
    --do_eval \
    --eval_checkpoint_path none \
    --use_retrieval_augmentation \
    --retrieval_concat_number 5 \
    --retrieval_index_path_for_train data/THG_twitter/twitter.2021.train.src_after_cleaning.txt_bm25_score.json \
    --retrieval_index_path_for_val data/THG_twitter/twitter.2021.valid.src_after_cleaning.txt_bm25_score.json \
    --retrieval_index_path_for_test data/THG_twitter/twitter.2021.test.src_after_cleaning.txt_bm25_score.json \
    --max_source_length 180 \
    --max_target_length 100 \
    --n_gpu 1 \
    --device cuda:0 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --num_train_epochs 10 \
    --seed 42 \
    --weight_decay 1e-5 \
    --adam_epsilon 1e-8 \
    --warmup_steps 0.06 \
    --load_pretrained_parameters
#    --do_direct_eval