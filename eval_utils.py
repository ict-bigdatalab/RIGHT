# -*- coding: utf-8 -*-
from Template import SEP
from rouge import Rouge
import copy


def extracte_hashtags_from_sequence(seq: str):
    seq = seq.strip()
    if seq == 'None':
        seq = ''
    hashtags = seq.split(SEP)
    hashtags = [ht.strip() for ht in hashtags]
    results = []
    for ht in hashtags:
        if ht != '' and ht not in results:
            results.append(ht)
    return results


def recall_k(y_pred, y_true, k=1):
    if k <= 0:
        raise ValueError(f"k must be greater than 0."
                         f"{k} received.")
    tp = 0.0
    total = 0
    true_list = copy.deepcopy(y_true)
    for i in range(len(y_pred)):
        total += len(y_true[i])
        for j in range(len(y_pred[i])):
            if j >= k:
                break
            if y_pred[i][j] in true_list[i]:
                tp += 1
                true_list[i].remove(y_pred[i][j])
    return tp / total


def precision_k(y_pred, y_true, k=1):
    if k <= 0:
        raise ValueError(f"k must be greater than 0."
                         f"{k} received.")
    tp = 0.0
    total = 0
    true_list = copy.deepcopy(y_true)
    for i in range(len(y_pred)):
        total += k
        for j in range(len(y_pred[i])):
            if j >= k:
                break
            if y_pred[i][j] in true_list[i]:
                tp += 1
                true_list[i].remove(y_pred[i][j])
    return tp / total


def f1(pre, rec):
    if pre == 0 and rec == 0:
        return 0.0
    return 2 * pre * rec / (pre + rec)


def compute_scores(out_seq, labels, language='en'):
    assert len(labels) == len(out_seq)
    if language == 'cn':
        f1_pre_hashtags = []
        f1_lab_hashtabs = []
        f5_pre_hashtags = []
        f5_lab_hashtabs = []
        for i in range(len(labels)):
            labels[i] = labels[i].strip()
            # labels[i] = (labels[i] + ' ' + SEP + ' ') * 5
            # labels[i] = labels[i][:(len(SEP) + 2) * (-1)].strip()
            out_seq[i] = out_seq[i].strip()
            f1_pre_hashtags.append([extracte_hashtags_from_sequence(out_seq[i])[0]])
            f1_lab_hashtabs.append(extracte_hashtags_from_sequence(labels[i]))
            f5_pre_hashtags.append(extracte_hashtags_from_sequence(out_seq[i]))
            f5_lab_hashtabs.append(extracte_hashtags_from_sequence(labels[i]) * 5)
            labels[i] = labels[i].replace(SEP, ' ')
            out_seq[i] = out_seq[i].split(SEP)[0].strip()
            labels[i] = labels[i].replace(' ', '').replace('', ' ').strip()
            out_seq[i] = out_seq[i].replace(' ', '').replace('', ' ').strip()
        rg = Rouge()
        rouge_score = rg.get_scores(out_seq, labels, avg=True)
        pre_1 = precision_k(f1_pre_hashtags, f1_lab_hashtabs, k=1)
        rec_1 = recall_k(f1_pre_hashtags, f1_lab_hashtabs, k=1)
        f1_1 = f1(pre_1, rec_1)
        pre_5 = precision_k(f5_pre_hashtags, f5_lab_hashtabs, k=5)
        rec_5 = recall_k(f5_pre_hashtags, f5_lab_hashtabs, k=5)
        f1_5 = f1(pre_5, rec_5)
    else:
        pre_hashtags = []
        lab_hashtabs = []
        for i in range(len(labels)):
            labels[i] = labels[i].strip()
            out_seq[i] = out_seq[i].strip()
            pre_hashtags.append(extracte_hashtags_from_sequence(out_seq[i]))
            lab_hashtabs.append(extracte_hashtags_from_sequence(labels[i]))
            labels[i] = labels[i].replace(SEP, ' ')
            out_seq[i] = out_seq[i].replace(SEP, ' ')
        rg = Rouge()
        rouge_score = rg.get_scores(out_seq, labels, avg=True)
        pre_1 = precision_k(pre_hashtags, lab_hashtabs, k=1)
        rec_1 = recall_k(pre_hashtags, lab_hashtabs, k=1)
        f1_1 = f1(pre_1, rec_1)
        pre_5 = precision_k(pre_hashtags, lab_hashtabs, k=5)
        rec_5 = recall_k(pre_hashtags, lab_hashtabs, k=5)
        f1_5 = f1(pre_5, rec_5)

    result = {
        'rouge': rouge_score,
        'precision_1': pre_1,
        'recall_1': rec_1,
        'f1_1': f1_1,
        'precision_5': pre_5,
        'recall_5': rec_5,
        'f1_5': f1_5,
    }
    return result


if __name__ == '__main__':
    # pre = ["a cat is on the table", 'a cat is on the table']
    # ref = ['there is a cat on the table', 'a cat is on the table']
    # rg = Rouge()
    # scores = rg.get_scores(pre, ref, avg=True)
    # print(scores)
    # a = "ac b <extra_id_0>www  "
    # print(extracte_hashtags_from_sequence(a))
    out_seq_path = 'outputs/THG/lr_3e-4_bs16_epoch10_simcsetunedretrieval_concattop9/test_output.txt'
    out_id = []
    out_seq = []
    out_label = []
    out_output = []
    with open(out_seq_path, 'r', encoding='UTF-8') as fp:
        for i in range(11328):
            id = int(fp.readline())
            input_seq = fp.readline()
            label = fp.readline()
            output = fp.readline()
            out_id.append(id)
            out_seq.append(input_seq)
            out_label.append(label)
            out_output.append(output)
    results = compute_scores(out_output, out_label)
    rouge_score = results['rouge']
    exp_results = f"test_rouge_1_p: {rouge_score['rouge-1']['p']};  test_rouge_1_r: {rouge_score['rouge-1']['r']};  test_rouge_1_f: {rouge_score['rouge-1']['f']} \n" \
                  f"test_rouge_2_p: {rouge_score['rouge-2']['p']};  test_rouge_2_r: {rouge_score['rouge-2']['r']};  test_rouge_2_f: {rouge_score['rouge-2']['f']} \n" \
                  f"test_rouge_l_p: {rouge_score['rouge-l']['p']};  test_rouge_l_r: {rouge_score['rouge-l']['r']};  test_rouge_l_f: {rouge_score['rouge-l']['f']} \n" \
                  f"test_precision_1: {results['precision_1']};  test_recall_1: {results['recall_1']};  test_f1_1: {results['f1_1']} \n" \
                  f"test_precision_5: {results['precision_5']};  test_recall_5: {results['recall_5']};  test_f1_5: {results['f1_5']} \n"
    print(exp_results)
