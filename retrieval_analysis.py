import json
from get_datasets import read_line_examples_from_file
from tqdm import tqdm
from eval_utils import f1


def get_hashtag_list(dst):
    tags = dst.split('[SEP]')
    target = []
    for j in range(len(tags)):
        tags[j] = tags[j].strip()
        if tags[j] != '':
            target.append(tags[j])
    # if the dst is nothing
    if len(target) == 0:
        target.append('None')
    # statistic_hashtags(hashtags)
    return target


def retrieval_analysis(src_path, label_path, rev_index_path, document_path, out_path):
    src_list = read_line_examples_from_file(src_path)
    dst_list = read_line_examples_from_file(label_path)
    document_list = read_line_examples_from_file(document_path)
    with open(rev_index_path, 'r', encoding='UTF-8') as fp:
        rev_index = json.load(fp)
    rev_dst = [[document_list[index] for index in rev_index[i]["index"]] for i in range(len(src_list))]

    with open(out_path, 'w', encoding='UTF-8') as fp:
        for i in tqdm(range(len(src_list))):
            line = str(i) + '\n' + src_list[i] + '\n' + dst_list[i] + '\n'
            for k in range(len(rev_dst[i])):
                line = line + str(rev_index[i]['score'][k]) + '\t' + rev_dst[i][k] + '\n'
            line += '\n'
            fp.write(line)


def retrieval_hashtag_score_analysis(src_path, label_path, rev_index_path, document_path, top_k):
    src_list = read_line_examples_from_file(src_path)
    dst_list = read_line_examples_from_file(label_path)
    document_list = read_line_examples_from_file(document_path)
    with open(rev_index_path, 'r', encoding='UTF-8') as fp:
        rev_index = json.load(fp)
    rev_dst = [[get_hashtag_list(document_list[index]) for index in rev_index[i]["index"]] for i in range(len(src_list))]
    dst_list = [get_hashtag_list(dst) for dst in dst_list]

    total_p = 0
    total_r = 0
    true_num = 0

    for i in tqdm(range(len(src_list))):
        label = dst_list[i]
        hashtag_score = dict()
        for k in range(len(rev_dst[i])):
            for rev_hashtag in rev_dst[i][k]:
                if rev_hashtag not in hashtag_score.keys():
                    hashtag_score[rev_hashtag] = 0
                hashtag_score[rev_hashtag] += rev_index[i]['score'][k]
        hashtag_score = sorted(hashtag_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        total_p += len(hashtag_score)
        total_r += len(label)
        for rev_hashtag_pair in hashtag_score:
            for lab in label:
                if rev_hashtag_pair[0] == lab or rev_hashtag_pair[0] in lab or lab in rev_hashtag_pair[0]:
                    true_num += 1
    p = true_num / total_p
    r = true_num / total_r
    f = f1(p, r)
    print(p)
    print(r)
    print(f)


if __name__ == '__main__':
    src_path = 'data/THG_twitter/twitter.2021.test.src_after_cleaning.txt'
    label_path = 'data/THG_twitter/twitter.2021.test.dst_after_cleaning.txt'
    rev_index_path = './data/THG_twitter/twitter.2021.test.src_after_cleaning.txt_bert_dense_score.json'
    document_path = 'data/THG_twitter/twitter.2021.train.dst_after_cleaning.txt'
    out_path = 'traing_bm25_retrieval_information'
    # retrieval_analysis(src_path, label_path, rev_index_path, document_path, out_path)
    retrieval_hashtag_score_analysis(src_path, label_path, rev_index_path, document_path, 4)