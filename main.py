# -*- coding: utf-8 -*-
import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup

from get_datasets import Twitter_THG
from eval_utils import compute_scores
from model import GenerativeModel

from Template import SEP

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--dataset", default="THG", type=str)
    parser.add_argument("--exp_version", default="default_test", type=str)
    parser.add_argument("--train_src_file", default='data/THG_twitter/twitter.2021.train.src_after_cleaning.txt', type=str,
                        help="The path of the training src dataset")
    parser.add_argument("--train_dst_file", default='data/THG_twitter/twitter.2021.train.dst_after_cleaning.txt', type=str,
                        help="The path of the training dst dataset")
    parser.add_argument("--val_src_file", default='data/THG_twitter/twitter.2021.valid.src_after_cleaning.txt', type=str,
                        help="The path of the validation src dataset")
    parser.add_argument("--val_dst_file", default='data/THG_twitter/twitter.2021.valid.dst_after_cleaning.txt', type=str,
                        help="The path of the validation dst dataset")
    parser.add_argument("--test_src_file", default='data/THG_twitter/twitter.2021.test.src_after_cleaning.txt', type=str,
                        help="The path of the test src dataset")
    parser.add_argument("--test_dst_file", default='data/THG_twitter/twitter.2021.test.dst_after_cleaning.txt', type=str,
                        help="The path of the test dst dataset")

    # parser.add_argument("--train_src_file", default='data/WHG/new4_train.src', type=str,
    #                     help="The path of the training src dataset")
    # parser.add_argument("--train_dst_file", default='data/WHG/new4_train.dst', type=str,
    #                     help="The path of the training dst dataset")
    # parser.add_argument("--val_src_file", default='data/WHG/new4_validation.src', type=str,
    #                     help="The path of the validation src dataset")
    # parser.add_argument("--val_dst_file", default='data/WHG/new4_validation.dst', type=str,
    #                     help="The path of the validation dst dataset")
    # parser.add_argument("--test_src_file", default='data/WHG/new4_test.src', type=str,
    #                     help="The path of the test src dataset")
    # parser.add_argument("--test_dst_file", default='data/WHG/new4_test.dst', type=str,
    #                     help="The path of the test dst dataset")

    # parser.add_argument("--train_src_file", default='data/THG_twitter/sample_src.txt',
    #                     type=str,
    #                     help="The path of the training src dataset")
    # parser.add_argument("--train_dst_file", default='data/THG_twitter/sample_dst.txt',
    #                     type=str,
    #                     help="The path of the training dst dataset")
    # parser.add_argument("--val_src_file", default='data/THG_twitter/sample_src.txt',
    #                     type=str,
    #                     help="The path of the validation src dataset")
    # parser.add_argument("--val_dst_file", default='data/THG_twitter/sample_dst.txt',
    #                     type=str,
    #                     help="The path of the validation dst dataset")
    # parser.add_argument("--test_src_file", default='data/THG_twitter/sample_src.txt',
    #                     type=str,
    #                     help="The path of the test src dataset")
    # parser.add_argument("--test_dst_file", default='data/THG_twitter/sample_dst.txt',
    #                     type=str,
    #                     help="The path of the test dst dataset")

    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--tokenizer_name_or_path", default='t5-base', type=str,
                        help="Path to tokenizer or shortcut name")
    # parser.add_argument("--model_name_or_path", default='google/mt5-small', type=str,
    #                     help="Path to pre-trained model or shortcut name")
    # parser.add_argument("--tokenizer_name_or_path", default='google/mt5-small', type=str,
    #                     help="Path to tokenizer or shortcut name")
    parser.add_argument("--load_pretrained_parameters", action='store_true', default=True,
                        help="Whether to load pretrained_parameters")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--eval_checkpoint_path", default="outputs/WHG/lr_3e-4_bs18_epoch10_seq2seq_baseline/final_model_dict.pkl", type=str,
                        help="The checkpoint path for directly testing. It works when do_direct_eval is True")
    # parser.add_argument("--do_inference", action='store_true',
    #                     help="Whether to run inference with trained checkpoints")

    # retrieval_augmentation
    parser.add_argument("--use_retrieval_augmentation", action='store_true', default=False,
                        help="Whether to use retrieval augmentation")
    parser.add_argument("--use_random_retrieval_augmentation", action='store_true', default=False,
                        help="Whether to use retrieval augmentation")
    parser.add_argument("--retrieval_index_path_for_train", default='data/THG_twitter/twitter.2021.train.src_after_cleaning.txt_simcse_tuned_dense_score.json', type=str,
                        help="The path of the retrieval index for training")
    parser.add_argument("--retrieval_index_path_for_val", default='data/THG_twitter/twitter.2021.valid.src_after_cleaning.txt_simcse_tuned_dense_score.json', type=str,
                        help="The path of the retrieval index for validation")
    parser.add_argument("--retrieval_index_path_for_test", default='data/THG_twitter/twitter.2021.test.src_after_cleaning.txt_simcse_tuned_dense_score.json', type=str,
                        help="The path of the retrieval index for testing")
    parser.add_argument("--retrieval_concat_number", default=5, type=int,
                        help="0 is concat all top_k retrieved hashtags. Other is the number of concat hashtags")

    # selector
    parser.add_argument("--use_selector_result", action='store_true', default=False,
                        help="Whether to use selector result")
    parser.add_argument("--selector_result_path_for_train",
                        default='', type=str,
                        help="The path of the retrieval index for training")
    parser.add_argument("--selector_result_path_for_val",
                        default='', type=str,
                        help="The path of the retrieval index for validation")
    parser.add_argument("--selector_result_path_for_test",
                        default='', type=str,
                        help="The path of the retrieval index for testing")
    
    parser.add_argument("--without_hashtag_ranking", action='store_true', default=False,
                        help="Whether to use selector result")

    # other parameters
    parser.add_argument("--max_source_length", default=180, type=int)
    parser.add_argument("--max_target_length", default=100, type=int)
    parser.add_argument("--n_gpu", default=0, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.06, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    output_dir = "outputs/" + args.dataset + '/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir += args.exp_version + '/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir
    return args


def get_dataset(tokenizer, args, mode):
    if mode in ['train', 'val', 'test']:
        return Twitter_THG(tokenizer, args, mode)
    else:
        raise ValueError("please give mode in [train, val, test]")


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams)
        print(self.hparams)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
        self.model = GenerativeModel(hparams, self.tokenizer)
        self.val_output_path = self.hparams.output_dir + 'val_output.txt'

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        # print("training_step")
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # print("training_epoch_end")
        # print(outputs)
        avg_train_loss = torch.Tensor([x["loss"] for x in outputs]).mean().item()
        self.log("avg_train_loss", avg_train_loss)
        self.print()
        self.print("training average loss: ", avg_train_loss)
        self.print()

    def validation_step(self, batch, batch_idx):
        # print("validation_step")
        # loss = self._step(batch)
        # self.log("val_loss", loss)
        sequences = self.model.generate(batch)  # num_beams=8, early_stopping=True)
        # print("outputs length: ", len(dec), "  inputs length: ", len(batch['source_seq']))
        return {"target_sentences": batch['target'], 'output_seq': sequences, "input_seq": batch['src']}

    def validation_epoch_end(self, outputs):
        out_seq = []
        labels = []
        input_seq = []
        for x in outputs:
            out_seq.extend(x['output_seq'])
            labels.extend(x['target_sentences'])
            input_seq.extend(x['input_seq'])
        with open(self.val_output_path, 'w') as f:
            for i in range(len(out_seq)):
                line = str(i) + "\n" + input_seq[i] + '\n' + labels[i] + '\n' + out_seq[i] + '\n'
                f.write(line)
        language = 'cn' if self.hparams.dataset == 'WHG' else 'en'
        result = compute_scores(out_seq, labels, language)
        rouge_score = result['rouge']
        self.log("val_rouge_1_p", rouge_score['rouge-1']['p'])
        self.log("val_rouge_1_r", rouge_score['rouge-1']['r'])
        self.log("val_rouge_1_f", rouge_score['rouge-1']['f'])
        self.log("val_rouge_2_p", rouge_score['rouge-2']['p'])
        self.log("val_rouge_2_r", rouge_score['rouge-2']['r'])
        self.log("val_rouge_2_f", rouge_score['rouge-2']['f'])
        self.log("val_rouge_L_p", rouge_score['rouge-l']['p'])
        self.log("val_rouge_L_r", rouge_score['rouge-l']['r'])
        self.log("val_rouge_L_f", rouge_score['rouge-l']['f'])

        self.log("val_precision_1", result['precision_1'])
        self.log("val_recall_1", result['recall_1'])
        self.log("val_f1_1", result['f1_1'])
        self.log("val_precision_5", result['precision_5'])
        self.log("val_recall_5", result['recall_5'])
        self.log("val_f1_5", result['f1_5'])

        self.log("val_rouge_average", (rouge_score['rouge-1']['f']+rouge_score['rouge-2']['f']+rouge_score['rouge-l']['f']+result['f1_1']+result['f1_5'])/5)

        self.print()
        self.print("val_rouge_1_f: ", rouge_score['rouge-1']['f'])
        self.print("val_rouge_2_f: ", rouge_score['rouge-2']['f'])
        self.print("val_rouge_L_f: ", rouge_score['rouge-l']['f'])
        self.print('val_f1_1: ', result['f1_1'])
        self.print('val_f1_5: ', result['f1_5'])
        self.print()

    def configure_optimizers(self):
        # print("configure_optimizers")
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None):
        # 下面这一行的这里面的参数可以去掉 但是不建议
        # print("optimizer_step")
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, args=self.hparams, mode='train')
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps * t_total, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, args=self.hparams, mode='val')
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


def evaluate(data_loader, model, args, tokenizer):
    """
    Compute scores given the predictions and gold labels
    """
    language = 'cn' if args.dataset == 'WHG' else 'en'
    model.eval()
    labels = []
    out_seq = []
    input_seq = []
    model.to(args.device)
    num_beam = 5 if args.dataset == 'WHG' else 1
    for batch in tqdm(data_loader):
        # need to push the data to device
        sequences = model.generate(batch, num_beams=num_beam)
        labels.extend(batch['target'])
        out_seq.extend(sequences)
        input_seq.extend(batch['src'])
    test_output_path = args.output_dir + 'test_output.txt'
    with open(test_output_path, 'w') as f:
        for i in range(len(out_seq)):
            line = str(i) + "\n" + input_seq[i] + '\n' + labels[i] + '\n' + out_seq[i] + '\n'
            f.write(line)
    result = compute_scores(out_seq, labels, language)
    return result


def main():
    args = init_args()
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    print("\n", "=" * 30, f"NEW EXP", "=" * 30, "\n")
    if args.n_gpu == 0:
        args.device = 'cpu'
    else:
        args.device = 'cuda:0'
    # training process
    if args.do_train:
        print("\n****** Conduct Training ******")
        # initialize the T5 model
        model = T5FineTuner(args)

        # prepare for trainer
        checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, monitor='val_rouge_average', filename='bestmodel_{epoch:02d}_{val_rouge_L_f:.4f}', mode="max")
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            # accelerator=args.device,
            # devices=args.n_gpu,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
            # limit_train_batches=0.5
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        torch.save(model.model.state_dict(), args.output_dir + "final_model_dict.pkl")
        print("Finish training and saving the model!")
        # print(checkpoint_callback.best_model_path)

    # evaluation
    if args.do_direct_eval or args.do_eval:
        print("\n****** Conduct Evaluating with the last state ******")
        print("Reload the model")

        if not args.do_direct_eval:
            args.eval_checkpoint_path = args.output_dir + "final_model_dict.pkl"

        # model = T5FineTuner(args, tfm_model, tokenizer)
        # model = T5FineTuner.load_from_checkpoint(
        #     'outputs/baseline_0/bestmodel_epoch=25_val_opt_f1=0.5250.ckpt', hparams_file='outputs/baseline_0/version_107546/hparams.yaml')
        # model = T5FineTuner.load_from_checkpoint(checkpoint_callback.best_model_path, hparams_file=args.output_dir + f'lightning_logs/version_0/hparams.yaml')
        model = GenerativeModel(args, tokenizer)
        model.load_state_dict(torch.load(args.eval_checkpoint_path))
        print("load: " + args.eval_checkpoint_path)
        test_dataset = get_dataset(tokenizer, args, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)

        # compute the performance scores
        results = evaluate(test_loader, model, args, tokenizer)
        rouge_score = results['rouge']
        # write to file
        log_file_path = f"{args.output_dir}/result_log.txt"
        local_time = time.asctime(time.localtime(time.time()))

        # exp_settings = f"Datset={args.dataset}; Exp setting={args.exp_version} Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}, model = {args.model_name_or_path}"
        exp_settings = ""
        for arg in vars(args):
            exp_settings = exp_settings + str(arg) + ":" + str(getattr(args, arg)) + "\n"
        exp_results = f"test_rouge_1_p: {rouge_score['rouge-1']['p']};  test_rouge_1_r: {rouge_score['rouge-1']['r']};  test_rouge_1_f: {rouge_score['rouge-1']['f']} \n" \
                      f"test_rouge_2_p: {rouge_score['rouge-2']['p']};  test_rouge_2_r: {rouge_score['rouge-2']['r']};  test_rouge_2_f: {rouge_score['rouge-2']['f']} \n" \
                      f"test_rouge_l_p: {rouge_score['rouge-l']['p']};  test_rouge_l_r: {rouge_score['rouge-l']['r']};  test_rouge_l_f: {rouge_score['rouge-l']['f']} \n" \
                      f"test_precision_1: {results['precision_1']};  test_recall_1: {results['recall_1']};  test_f1_1: {results['f1_1']} \n" \
                      f"test_precision_5: {results['precision_5']};  test_recall_5: {results['recall_5']};  test_f1_5: {results['f1_5']} \n"
        log_str = f'============================================================\n'
        log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

        print(log_str)
        with open(log_file_path, "a+") as f:
            f.write(log_str)
        print("Finish test!")


if __name__ == '__main__':
    main()
    # args = init_args()
    # tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    # model = GenerativeModel(args, tokenizer)
    # model.load_state_dict(torch.load(args.eval_checkpoint_path))
    # datasets = get_dataset(tokenizer, args, 'train')
    # print(datasets[0])
    # print(datasets[1])
