#!/bin/bash
python3 main.py --dataset WY \
    --exp_version lr_1e-4_bs12_epoch30_seq2seq_baseline \
    --train_src_file data/Twitter_naacl_wangyue/train_src.txt \
    --train_dst_file data/Twitter_naacl_wangyue/train_dst.txt \
    --val_src_file data/Twitter_naacl_wangyue/valid_src.txt \
    --val_dst_file data/Twitter_naacl_wangyue/valid_dst.txt \
    --test_src_file data/Twitter_naacl_wangyue/test_src.txt \
    --test_dst_file data/Twitter_naacl_wangyue/test_dst.txt \
    --model_name_or_path t5-base \
    --tokenizer_name_or_path t5-base \
    --do_train \
    --do_eval \
    --eval_checkpoint_path none \
    --retrieval_index_path_for_train none \
    --retrieval_index_path_for_val none \
    --retrieval_index_path_for_test none \
    --max_source_length 256 \
    --max_target_length 32 \
    --n_gpu 1 \
    --device cuda:0 \
    --train_batch_size 12 \
    --eval_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 30 \
    --seed 42 \
    --weight_decay 1e-5 \
    --adam_epsilon 1e-8 \
    --warmup_steps 0.06
#    --use_retrieval_augmentation
#    --do_direct_eval