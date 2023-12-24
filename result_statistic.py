import json
from get_datasets import read_line_examples_from_file
from tqdm import tqdm
from eval_utils import f1


def get_hashtag_list(dst):
    tags = dst.split('<extra_id_0>')
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


def statistic_result_after_retrieval_augmentation(baseline_result_path, after_result_path):
    number = 11327
    number = 4454
    baseline_id = []
    baseline_seq = []
    baseline_label = []
    baseline_output = []
    with open(baseline_result_path, 'r', encoding='UTF-8') as fp:
        for i in range(number):
            id = int(fp.readline())
            input_seq = fp.readline()
            label = fp.readline()
            output = fp.readline()
            baseline_id.append(id)
            baseline_seq.append(input_seq)
            baseline_label.append(get_hashtag_list(label))
            baseline_output.append(get_hashtag_list(output))

    after_id = []
    after_seq = []
    after_label = []
    after_output = []
    with open(after_result_path, 'r', encoding='UTF-8') as fp:
        for i in range(number):
            id = int(fp.readline())
            input_seq = fp.readline()
            label = fp.readline()
            output = fp.readline()
            after_id.append(id)
            after_seq.append(input_seq)
            after_label.append(get_hashtag_list(label))
            after_output.append(get_hashtag_list(output))

    num_hushtag_bT_aF_in = 0
    num_hashtag_bF_aT_in = 0
    num_hashtag_bT_aF_out = 0
    num_hashtag_bF_aT_out = 0
    total_num_hashtag = 0

    num_case_bT_aF_in = 0
    num_case_bF_aT_in = 0
    num_case_bT_aF_out = 0
    num_case_bF_aT_out = 0
    total_num_case = 0

    for i in range(number):
        reg_arg_seq = after_seq[i]
        retrieval_hashtag = reg_arg_seq.split('<extra_id_1>')[1:]
        retrieval_hashtag = [tag.strip() for tag in retrieval_hashtag]
        label = after_label[i]
        bs_out = baseline_output[i]
        af_out = after_output[i]
        hushtag_bT_aF_in = 0
        hashtag_bF_aT_in = 0
        hashtag_bT_aF_out = 0
        hashtag_bF_aT_out = 0
        for lab in label:
            total_num_hashtag += 1
            if lab in bs_out:
                if lab not in af_out:
                    if lab in retrieval_hashtag:
                        hushtag_bT_aF_in += 1
                    else:
                        hashtag_bT_aF_out += 1
            else:
                if lab in af_out:
                    if lab in retrieval_hashtag:
                        hashtag_bF_aT_in += 1
                    else:
                        hashtag_bF_aT_out += 1
        num_hushtag_bT_aF_in += hushtag_bT_aF_in
        num_hashtag_bF_aT_in += hashtag_bF_aT_in
        num_hashtag_bT_aF_out += hashtag_bT_aF_out
        num_hashtag_bF_aT_out += hashtag_bF_aT_out

        total_num_case += 1
        if hushtag_bT_aF_in > 0:
            num_case_bT_aF_in += 1
        if hashtag_bT_aF_out > 0:
            num_case_bT_aF_out += 1
        if hashtag_bF_aT_in > 0:
            num_case_bF_aT_in += 1
        if hashtag_bF_aT_out > 0:
            num_case_bF_aT_out += 1

    probability_hashtag_bT_aF_in = num_hushtag_bT_aF_in / total_num_hashtag
    probability_hashtag_bF_aT_in = num_hashtag_bF_aT_in / total_num_hashtag
    probability_hashtag_bT_aF_out = num_hashtag_bT_aF_out / total_num_hashtag
    probability_hashtag_bF_aT_out = num_hashtag_bF_aT_out / total_num_hashtag

    probability_case_bT_aF_in = num_case_bT_aF_in / total_num_case
    probability_case_bF_aT_in = num_case_bF_aT_in / total_num_case
    probability_case_bT_aF_out = num_case_bT_aF_out / total_num_case
    probability_case_bF_aT_out = num_case_bF_aT_out / total_num_case

    print("="*20, "Hashtag", "="*20)
    print(f"Hashtag is True in baseline but False in afterModel and in Retrieval: num is {num_hushtag_bT_aF_in} \t "
          f"probability is {probability_hashtag_bT_aF_in * 100}")
    print(f"Hashtag is False in baseline but True in afterModel and in Retrieval: num is {num_hashtag_bF_aT_in} \t "
          f"probability is {probability_hashtag_bF_aT_in * 100}")
    print(f"Hashtag is True in baseline but False in afterModel and NOT in Retrieval: num is {num_hashtag_bT_aF_out} \t "
          f"probability is {probability_hashtag_bT_aF_out * 100}")
    print(f"Hashtag is False in baseline but True in afterModel and NOT in Retrieval: num is {num_hashtag_bF_aT_out} \t "
          f"probability is {probability_hashtag_bF_aT_out * 100}")

    print("=" * 20, "Case", "=" * 20)
    print(f"Case is True in baseline but False in afterModel and in Retrieval: num is {num_case_bT_aF_in} \t "
          f"probability is {probability_case_bT_aF_in * 100}")
    print(f"Case is False in baseline but True in afterModel and in Retrieval: num is {num_case_bF_aT_in} \t "
          f"probability is {probability_case_bF_aT_in * 100}")
    print(
        f"Case is True in baseline but False in afterModel and NOT in Retrieval: num is {num_case_bT_aF_out} \t "
        f"probability is {probability_case_bT_aF_out * 100}")
    print(
        f"Case is False in baseline but True in afterModel and NOT in Retrieval: num is {num_case_bF_aT_out} \t "
        f"probability is {probability_case_bF_aT_out * 100}")


if __name__ == '__main__':
    baseline_result_path_THG_baseline = 'outputs/THG/lr_3e-4_bs48_epoch30_cleaning/test_output.txt'
    after_result_path_THG_cancatall = 'outputs/THG/lr_3e-4_bs32_epoch30_bm25retrieval_concatall/test_output.txt'

    baseline_result_path_WY_baseline = 'outputs/WY/lr_1e-4_bs12_epoch30_seq2seq_baseline/test_output.txt'
    after_result_path_WY_cancatall = 'outputs/WY/lr_1e-4_bs12_epoch30_seq2seq_bm25_concatall/test_output.txt'
    statistic_result_after_retrieval_augmentation(baseline_result_path_WY_baseline, after_result_path_WY_cancatall)