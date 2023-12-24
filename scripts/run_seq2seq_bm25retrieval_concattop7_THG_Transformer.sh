#!/bin/bash
python3 main.py --dataset THG \
    --exp_version lr_5e-5_bs96_epoch30_bm25retrieval_concattop7_Transformer \
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
    --retrieval_concat_number 7 \
    --retrieval_index_path_for_train data/THG_twitter/twitter.2021.train.src_after_cleaning.txt_bm25_score.json \
    --retrieval_index_path_for_val data/THG_twitter/twitter.2021.valid.src_after_cleaning.txt_bm25_score.json \
    --retrieval_index_path_for_test data/THG_twitter/twitter.2021.test.src_after_cleaning.txt_bm25_score.json \
    --max_source_length 256 \
    --max_target_length 100 \
    --n_gpu 1 \
    --device cuda:0 \
    --train_batch_size 96 \
    --eval_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 30 \
    --seed 42 \
    --weight_decay 1e-5 \
    --adam_epsilon 1e-6 \
    --warmup_steps 0.04
#    --load_pretrained_parameters
#    --do_direct_eval