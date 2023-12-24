import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration
import argparse

from get_datasets import Twitter_THG
from torch.utils.data import DataLoader
from Template import SEP, MAP_SPETOKENS_IDS
from eval_utils import extracte_hashtags_from_sequence


class GenerativeModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        if args.dataset == 'THG':
            if args.load_pretrained_parameters:
                self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name_or_path)
                print(f"\nthe model is {self.args.model_name_or_path} with pretrained parameters")
            else:
                config = T5Config.from_pretrained(self.args.model_name_or_path)
                self.model = AutoModelForSeq2SeqLM.from_config(config)
                print(f"\nthe model is {self.args.model_name_or_path} from scratch")
        elif args.dataset == 'WHG':
            self.model = MT5ForConditionalGeneration.from_pretrained(self.args.model_name_or_path)
            print(f"\nthe model is {self.args.model_name_or_path} with pretrained parameters")

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             labels=labels)
        return outputs

    def generate(self, batch, num_beams=1):
        self.eval()
        if self.args.dataset == 'WHG':
            with torch.no_grad():
                outputs = self.model.generate(batch['source_ids'].to(self.args.device),
                                              attention_mask=batch['source_mask'].to(self.args.device),
                                              num_beams=num_beams,
                                              max_length=self.args.max_target_length,
                                              num_return_sequences=num_beams
                                              )
                decs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
                dec = []
                batch_size = len(batch['src'])
                for bs in range(batch_size):
                    hashtag_str = ''
                    for d in range(bs * num_beams, (bs+1) * num_beams, 1):
                        hashtag_str = hashtag_str + decs[d] + ' ' + SEP + ' '
                    hashtag_str = hashtag_str[:(len(SEP) + 2) * (-1)].strip()
                    dec.append(hashtag_str)
        else:
            with torch.no_grad():
                # if num_beams == 1:
                #     self.model._cache_input_ids = batch['source_ids'].to(self.args.device)
                # else:
                #     expanded_return_idx = (
                #         torch.arange(batch['source_ids'].shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(
                #             self.to(self.args.device))
                #     )
                #     input_ids = batch['source_ids'].index_select(0, expanded_return_idx)
                #     self.model._cache_input_ids = input_ids.to(self.args.device)

                outputs = self.model.generate(batch['source_ids'].to(self.args.device),
                                              attention_mask=batch['source_mask'].to(self.args.device),
                                              num_beams=num_beams,
                                              max_length=self.args.max_target_length,
                                              )
            # decode outputs
            sequences = outputs
            dec = [self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids in
                   sequences]
            for d in range(len(dec)):
                dec[d] = dec[d].replace('<pad>', '')
                dec[d] = dec[d].replace('</s>', '').strip()
                result = extracte_hashtags_from_sequence(dec[d])
                dec[d] = ""
                if len(result) == 0:
                    dec[d] = "None"
                else:
                    for res in result:
                        dec[d] = dec[d] + res + " " + SEP + " "
                    dec[d] = dec[d][:(len(SEP) + 2) * (-1)].strip()
        self.train()
        # the shape is [batch_size, seq_len]
        return dec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="./PLM_checkpoint/t5-base", type=str)
    parser.add_argument("--device", default='cpu', type=str,)
    parser.add_argument("--max_target_length", default=100, type=int)
    args = parser.parse_args()
    tokenizer = T5Tokenizer.from_pretrained('PLM_checkpoint/t5-base')
    model = GenerativeModel(args, tokenizer)
    src_path = 'data/THG_twitter/twitter.2021.valid.src'
    dst_path = 'data/THG_twitter/twitter.2021.valid.dst'
    datasets = Twitter_THG(tokenizer, src_path, dst_path)
    data = DataLoader(datasets, 2, False)
    for batch in data:
        print(model.generate(batch))
        break
