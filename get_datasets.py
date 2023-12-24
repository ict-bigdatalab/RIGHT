import torch
from torch.utils.data import Dataset
import json
from Template import SEP, RETRIEVAL_LABEL_SEP, MAP_SPETOKENS_IDS
import matplotlib.pyplot as plt
from transformers import T5Tokenizer
from tqdm import tqdm
import random


def get_hashtag_list(dst):
    tags = dst.split('[SEP]')
    target = []
    for j in range(len(tags)):
        tags[j] = tags[j].strip()
        if tags[j] != '' and tags[j] not in target:
            target.append(tags[j])
    # if the dst is nothing
    if len(target) == 0:
        target.append('None')
    # statistic_hashtags(hashtags)
    return target


def read_line_examples_from_file(data_path):
    sequence = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            sequence.append(line.strip())
    return sequence


def statistic_hashtags(hashtags):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    total_num = len(hashtags)
    max_num = 0
    sum_num = 0
    nums = []
    max_len = 0
    sum_len = 0
    lens = []
    for hashtag_list in tqdm(hashtags):
        num = len(hashtag_list)
        max_num = max(num, max_num)
        sum_num += num
        nums.append(num)
        for hashtag in hashtag_list:
            tokens = tokenizer(hashtag)['input_ids']
            llen = len(tokens) - 1
            max_len = max(llen, max_len)
            sum_len += llen
            lens.append(llen)
    avg_num = sum_num / total_num
    avg_len = sum_len / sum_num
    top_num = max_num + 1
    top_len = max_len + 1

    bar_list_num = [0] * top_num
    bar_list_len = [0] * top_len
    for n in nums:
        bar_list_num[n] += 1
    for le in lens:
        bar_list_len[le] += 1

    for i in range(top_num):
        plt.bar(i, bar_list_num[i])
    plt.title('hashtag num per post')
    plt.xlabel("num")
    plt.ylabel("Number")
    plt.show()

    for i in range(top_len):
        plt.bar(i, bar_list_len[i])
    plt.title('hashtag length')
    plt.xlabel("length")
    plt.ylabel("Number")
    plt.show()

    print(f"number of hashtags per post:\n max num is {max_num} \n average num is {avg_num} \n nums statistics: {bar_list_num}\n")
    print(f"length of hashtag:\n max length is {max_len} \n average length is {avg_len} \n nums statistics: {bar_list_len}\n")


def get_para_targets(dst):
    targets = []
    hashtags = []
    for i in range(len(dst)):
        tags = dst[i].split('[SEP]')
        target = []
        for j in range(len(tags)):
            tags[j] = tags[j].strip()
            if tags[j] != '':
                target.append(tags[j])
        # if the dst is nothing
        if len(target) == 0:
            target.append('None')
        hashtags.append(target)
        seq = ""
        for t in target:
            seq = seq + t + " " + SEP + " "
        seq = seq[:(len(SEP)+2) * (-1)].strip()
        targets.append(seq)
    # statistic_hashtags(hashtags)
    return targets, hashtags


def get_transformed_io(src_path, dst_path):
    src = read_line_examples_from_file(src_path)
    dst = read_line_examples_from_file(dst_path)
    assert len(src) == len(dst)
    print(f"Total examples = {len(dst)}")
    targets, hashtags = get_para_targets(dst)
    assert len(src) == len(targets)
    return src, targets, hashtags


def vision_dataLen_distribution(data, tokenizer, title=""):
    tokens = tokenizer([i for i in data])
    maxlen = 0
    sum_len = 0
    for i in tokens["input_ids"]:
        sum_len += (len(i)-1)
        if len(i) > maxlen:
            maxlen = len(i)
    maxlen += 1
    lenn = [0] * maxlen
    for i in tokens["input_ids"]:
        lenn[len(i)] += 1
    for i in range(maxlen):
        plt.bar(i, lenn[i])
    plt.title(title)
    plt.xlabel("Length")
    plt.ylabel("Number")
    plt.show()
    avg_len = sum_len / len(tokens["input_ids"])
    print("avg_len: ", avg_len)


def retrieval_augment(args, src, retrieval_index_path, retrieval_document_path, selector_result_path, top_k):
    with open(retrieval_index_path, 'r', encoding='UTF-8') as fp:
        retrieval_index = json.load(fp)
    if selector_result_path:
        with open(selector_result_path, 'r', encoding='UTF-8') as fp:
            selector_result = json.load(fp)
    retrieval_document = read_line_examples_from_file(retrieval_document_path)
    assert len(src) == len(retrieval_index)
    rev_dst = [[get_hashtag_list(retrieval_document[index]) for index in retrieval_index[i]["index"]] for i in
               range(len(src))]
    for i, s in enumerate(src):
        if not args.use_random_retrieval_augmentation:
            if args.retrieval_concat_number == 0:
                retrieval_doc = ""
                tok_k_doc = [retrieval_document[retrieval_index[i]["index"][kk]].replace('[SEP]', RETRIEVAL_LABEL_SEP) for kk in range(top_k)]
                for doc in tok_k_doc:
                    retrieval_doc = retrieval_doc + " " + RETRIEVAL_LABEL_SEP + " " + doc
            else:
                retrieval_doc = ""
                if not args.use_selector_result:
                    hashtag_score = dict()
                    for k in range(len(rev_dst[i])):
                        for rev_hashtag in rev_dst[i][k]:
                            if rev_hashtag not in hashtag_score.keys():
                                hashtag_score[rev_hashtag] = 0
                            hashtag_score[rev_hashtag] += retrieval_index[i]['score'][k]
                    if args.without_hashtag_ranking:
                        random_index = []
                        augment_number = args.retrieval_concat_number
                        if len(hashtag_score) < args.retrieval_concat_number:
                            augment_number = len(hashtag_score)
                        while len(random_index) < augment_number:
                            rd_idx = random.randint(0, len(hashtag_score)-1)
                            if rd_idx not in random_index:
                                random_index.append(rd_idx)
                        hashtag_score = [list(hashtag_score.items())[rd_idx] for rd_idx in random_index]
                    else:
                        hashtag_score = sorted(hashtag_score.items(), key=lambda x: x[1], reverse=True)[:args.retrieval_concat_number]
                    for hashtag in hashtag_score:
                        retrieval_doc = retrieval_doc + " " + RETRIEVAL_LABEL_SEP + " " + hashtag[0]
                else:
                    hashtag_score = dict()
                    hashtag_num = dict()
                    for k in range(len(rev_dst[i])):
                        for rev_hashtag in rev_dst[i][k]:
                            if rev_hashtag not in hashtag_score.keys():
                                hashtag_score[rev_hashtag] = 0
                                hashtag_num[rev_hashtag] = 0
                            hashtag_score[rev_hashtag] += retrieval_index[i]['score'][k]
                            hashtag_num[rev_hashtag] += 1
                    for rev_hashtag in hashtag_score.keys():
                        hashtag_score[rev_hashtag] /= hashtag_num[rev_hashtag]
                        hashtag_score[rev_hashtag] += selector_result[i][rev_hashtag]
                        hashtag_score[rev_hashtag] = hashtag_score[rev_hashtag] * (1 + ((hashtag_num[rev_hashtag]-1) / 10))
                    hashtag_score = sorted(hashtag_score.items(), key=lambda x: x[1], reverse=True)[
                                    :args.retrieval_concat_number]
                    for hashtag in hashtag_score:
                        retrieval_doc = retrieval_doc + " " + RETRIEVAL_LABEL_SEP + " " + hashtag[0]
        else:
            retrieval_doc = ""
            random_index = []
            while len(random_index) < args.retrieval_concat_number:
                rd_idx = random.randint(0, len(retrieval_document)-1)
                if rd_idx not in random_index and rd_idx != i:
                    random_index.append(rd_idx)
            tok_k_doc = [retrieval_document[kk].split('[SEP]')[0].strip() for
                         kk in random_index]
            for doc in tok_k_doc:
                retrieval_doc = retrieval_doc + " " + RETRIEVAL_LABEL_SEP + " " + doc
        src[i] = src[i] + ' ' + retrieval_doc
    return src


class Twitter_THG(Dataset):
    def __init__(self, tokenizer, args, mode):
        super(Twitter_THG, self).__init__()
        if mode == 'train':
            self.src_data_path = args.train_src_file
            self.dst_data_path = args.train_dst_file
        elif mode == 'val':
            self.src_data_path = args.val_src_file
            self.dst_data_path = args.val_dst_file
        elif mode == 'test':
            self.src_data_path = args.test_src_file
            self.dst_data_path = args.test_dst_file
        else:
            raise ValueError("please give mode in [train, val, test]")

        if args.use_retrieval_augmentation:
            if mode == 'train':
                self.retrieval_index_path = args.retrieval_index_path_for_train
            elif mode == 'val':
                self.retrieval_index_path = args.retrieval_index_path_for_val
            elif mode == 'test':
                self.retrieval_index_path = args.retrieval_index_path_for_test
            self.retrieval_document_path = args.train_dst_file
            self.selector_result_path = None
            if args.use_selector_result:
                if mode == 'train':
                    self.selector_result_path = args.selector_result_path_for_train
                elif mode == 'val':
                    self.selector_result_path = args.selector_result_path_for_val
                elif mode == 'test':
                    self.selector_result_path = args.selector_result_path_for_test
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        self.args = args

        self.inputs = []
        self.targets = []
        self._build_datasets()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "src": self.src[index], "target": self.tar_seq[index]}

    def _build_datasets(self):
        src, targets, hashtags = get_transformed_io(self.src_data_path, self.dst_data_path)
        if self.args.use_retrieval_augmentation:
            src = retrieval_augment(self.args, src, self.retrieval_index_path, self.retrieval_document_path, self.selector_result_path, top_k=5)
        self.src = src
        self.tar_seq = targets
        # vision_dataLen_distribution(src, self.tokenizer, "input text distribution")
        # vision_dataLen_distribution(targets, self.tokenizer, "output text distribution")
        for i in range(len(src)):
            input = src[i]
            tokenized_input = self.tokenizer(
                [input], max_length=self.max_source_length, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_input)

            target = targets[i]
            tokenized_target = self.tokenizer(
                [target], max_length=self.max_target_length, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            self.targets.append(tokenized_target)


if __name__ == '__main__':
    # src_path = 'data/THG_twitter/twitter.2021.test.src_after_cleaning.txt'
    # dst_path = 'data/THG_twitter/twitter.2021.test.dst_after_cleaning.txt'
    tokenizers = T5Tokenizer.from_pretrained('PLM_checkpoint/t5-base')
    # datasets = Twitter_THG(tokenizers, src_path, dst_path)
    # get_transformed_io(src_path, dst_path)
    hashtags = ['da 19 eu', 'bucharest', 'ela', 'bratislava', 'eu', 'da 19 eu bucharest ela bratislava']