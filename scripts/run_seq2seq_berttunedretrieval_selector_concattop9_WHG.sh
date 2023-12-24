#!/bin/bash
python3 main.py --dataset WHG \
    --exp_version lr_3e-4_bs18_epoch10_berttunedretrieval_selector_concattop9 \
    --train_src_file data/WHG/new4_train.src_after_cleaning.txt \
    --train_dst_file data/WHG/new4_train.dst_after_cleaning.txt \
    --val_src_file data/WHG/new4_validation.src \
    --val_dst_file data/WHG/new4_validation.dst \
    --test_src_file data/WHG/new4_test.src \
    --test_dst_file data/WHG/new4_test.dst \
    --model_name_or_path google/mt5-small \
    --tokenizer_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --eval_checkpoint_path none \
    --use_retrieval_augmentation \
    --retrieval_concat_number 9 \
    --retrieval_index_path_for_train data/WHG/new4_train.src_after_cleaning.txt_bert_dense_score.json \
    --retrieval_index_path_for_val data/WHG/new4_validation.src_bert_dense_score.json \
    --retrieval_index_path_for_test data/WHG/new4_test.src_bert_dense_score.json \
    --use_selector_result \
    --selector_result_path_for_train data/WHG/train_tunedbert_selector_result.json \
    --selector_result_path_for_val data/WHG/validation_tunedbert_selector_result.json \
    --selector_result_path_for_test data/WHG/test_tunedbert_selector_result.json \
    --max_source_length 256 \
    --max_target_length 32 \
    --n_gpu 1 \
    --device cuda:0 \
    --train_batch_size 18 \
    --eval_batch_size 72 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --num_train_epochs 10 \
    --seed 42 \
    --weight_decay 1e-5 \
    --adam_epsilon 1e-8 \
    --warmup_steps 0.06 \
    --load_pretrained_parameters
#    --do_direct_eval