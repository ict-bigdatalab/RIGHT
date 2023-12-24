import json
from tqdm import tqdm
from gensim import corpora
from gensim.summarization.bm25 import BM25
from nltk.corpus import stopwords
from transformers import BertModel
from transformers import BertTokenizer
from dense_retrieval import MySimCSE
from get_datasets import get_transformed_io, get_hashtag_list
from eval_utils import f1
from functools import cmp_to_key
import csv
from data_augmentation import random_augmentation
import jieba
import jieba.posseg as pseg


def generate_index_json_file(data_path):
    out_path = data_path + "_index.json"
    data = []
    i = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip("\n")
            if not line:
                continue
            item = {
                "id": i,
                "contents": line
            }
            data.append(item)
            i += 1
            print(i)
    jsontext = json.dumps(data, indent=4)
    with open(out_path, 'w') as json_file:
        json_file.write(jsontext)


def bm25_retrieval_results(train_src_data_path, val_src_data_path, test_src_data_path):
    train_out_path = train_src_data_path + "_bm25_index.json"
    val_out_path = val_src_data_path + "_bm25_index.json"
    test_out_path = test_src_data_path + "_bm25_index.json"
    # read training documents
    documents = []
    with open(train_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            documents.append(line)
    print("The number of training documents is: ", len(documents))

    # read val queries
    val_queries = []
    with open(val_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            val_queries.append(line)
    print("The number of val queries is: ", len(val_queries))

    # read test queries
    test_queries = []
    with open(test_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            test_queries.append(line)
    print("The number of test queries is: ", len(test_queries))

    # build document index
    # split word
    texts = [doc.split() for doc in documents]
    # remove stopwords
    for i in range(len(texts)):
        texts[i] = [word for word in texts[i] if word not in stopwords.words('english')]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    bm25_obj = BM25(corpus)

    # training query
    train_query_result = []
    print("Start to create training queries result...")
    for i in tqdm(range(len(documents))):
        query = texts[i]
        query_doc = dictionary.doc2bow(query)
        scores = bm25_obj.get_scores(query_doc)
        best_docs = sorted(range(len(scores)), key=lambda j: scores[j])[-11:][::-1]
        if i in best_docs:
            best_docs.remove(i)
        else:
            best_docs = best_docs[:10]
            print(documents[i])
        train_query_result.append(best_docs)
    json_str = json.dumps(train_query_result, indent=4)
    with open(train_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating training queries result!")

    # val query
    val_query_result = []
    print("Start to create val queries result...")
    val_texts = [vq.split() for vq in val_queries]
    for i in tqdm(range(len(val_texts))):
        query = [word for word in val_texts[i] if word not in stopwords.words('english')]
        query_doc = dictionary.doc2bow(query)
        scores = bm25_obj.get_scores(query_doc)
        best_docs = sorted(range(len(scores)), key=lambda j: scores[j])[-10:][::-1]
        val_query_result.append(best_docs)
    json_str = json.dumps(val_query_result, indent=4)
    with open(val_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating val queries result!")

    # test query
    test_query_result = []
    print("Start to create test queries result...")
    test_texts = [tq.split() for tq in test_queries]
    for i in tqdm(range(len(test_texts))):
        query = [word for word in test_texts[i] if word not in stopwords.words('english')]
        query_doc = dictionary.doc2bow(query)
        scores = bm25_obj.get_scores(query_doc)
        best_docs = sorted(range(len(scores)), key=lambda j: scores[j])[-10:][::-1]
        test_query_result.append(best_docs)
    json_str = json.dumps(test_query_result, indent=4)
    with open(test_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating test queries result!")

def bm25_retrieval_score_results(train_src_data_path, val_src_data_path, test_src_data_path):
    train_out_path = train_src_data_path + "_bm25_score.json"
    val_out_path = val_src_data_path + "_bm25_score.json"
    test_out_path = test_src_data_path + "_bm25_score.json"
    # read training documents
    documents = []
    with open(train_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            documents.append(line)
    print("The number of training documents is: ", len(documents))

    # read val queries
    val_queries = []
    with open(val_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            val_queries.append(line)
    print("The number of val queries is: ", len(val_queries))

    # read test queries
    test_queries = []
    with open(test_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            test_queries.append(line)
    print("The number of test queries is: ", len(test_queries))

    # build document index
    # split word
    texts = [doc.split() for doc in documents]
    # remove stopwords
    for i in range(len(texts)):
        texts[i] = [word for word in texts[i] if word not in stopwords.words('english')]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    bm25_obj = BM25(corpus)

    # training query
    train_query_result = []
    print("Start to create training queries result...")
    for i in tqdm(range(len(documents))):
        query = texts[i]
        query_doc = dictionary.doc2bow(query)
        scores = bm25_obj.get_scores(query_doc)
        best_docs = sorted(range(len(scores)), key=lambda j: scores[j])[-11:][::-1]
        if i in best_docs:
            best_docs.remove(i)
        else:
            best_docs = best_docs[:10]
            print(documents[i])
        train_query_item = dict()
        train_query_item['index'] = best_docs
        train_query_item['score'] = [scores[doc] for doc in best_docs]
        train_query_result.append(train_query_item)
    json_str = json.dumps(train_query_result, indent=4)
    with open(train_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating training queries result!")

    # val query
    val_query_result = []
    print("Start to create val queries result...")
    val_texts = [vq.split() for vq in val_queries]
    for i in tqdm(range(len(val_texts))):
        query = [word for word in val_texts[i] if word not in stopwords.words('english')]
        query_doc = dictionary.doc2bow(query)
        scores = bm25_obj.get_scores(query_doc)
        best_docs = sorted(range(len(scores)), key=lambda j: scores[j])[-10:][::-1]
        val_query_item = dict()
        val_query_item['index'] = best_docs
        val_query_item['score'] = [scores[doc] for doc in best_docs]
        val_query_result.append(val_query_item)
    json_str = json.dumps(val_query_result, indent=4)
    with open(val_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating val queries result!")

    # test query
    test_query_result = []
    print("Start to create test queries result...")
    test_texts = [tq.split() for tq in test_queries]
    for i in tqdm(range(len(test_texts))):
        query = [word for word in test_texts[i] if word not in stopwords.words('english')]
        query_doc = dictionary.doc2bow(query)
        scores = bm25_obj.get_scores(query_doc)
        best_docs = sorted(range(len(scores)), key=lambda j: scores[j])[-10:][::-1]
        test_query_item = dict()
        test_query_item['index'] = best_docs
        test_query_item['score'] = [scores[doc] for doc in best_docs]
        test_query_result.append(test_query_item)
    json_str = json.dumps(test_query_result, indent=4)
    with open(test_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating test queries result!")


def dense_retrieval_results(train_src_data_path, val_src_data_path, test_src_data_path):
    train_out_path = train_src_data_path + "_bert_original_score.json"
    val_out_path = val_src_data_path + "_bert_original_score.json"
    test_out_path = test_src_data_path + "_bert_original_score.json"

    # loading model
    # model = MySimCSE("princeton-nlp/sup-simcse-roberta-large", device='cpu')
    # model = MySimCSE("/home/qiupeng/frz_project/SimCSE/result/retrieval_bert_chinese_base", device='cuda', pooler='cls')
    model = MySimCSE("bert-base-chinese", device='cuda', pooler='cls')

    # read training documents
    documents = []
    with open(train_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            documents.append(line)
    print("The number of training documents is: ", len(documents))

    # read val queries
    val_queries = []
    with open(val_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            val_queries.append(line)
    print("The number of val queries is: ", len(val_queries))

    # read test queries
    test_queries = []
    with open(test_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            test_queries.append(line)
    print("The number of test queries is: ", len(test_queries))

    # create index
    model.build_index(documents, device='cuda', batch_size=64)

    # training query
    train_query_result = []
    print("Start to create training queries result...")
    for i in tqdm(range(len(documents))):
        results = model.search(documents[i], device='cuda', threshold=-99, top_k=11)
        for k in range(len(results)):
            if results[k][0] == i:
                results.pop(k)
                break
        if len(results) > 10:
            results = results[:10]
        train_query_item = dict()
        train_query_item['index'] = [ind for ind, score in results]
        train_query_item['score'] = [score for ind, score in results]
        train_query_result.append(train_query_item)
    json_str = json.dumps(train_query_result, indent=4)
    with open(train_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating training queries result!")

    # val query
    val_query_result = []
    print("Start to create val queries result...")
    for i in tqdm(range(len(val_queries))):
        query = val_queries[i]
        results = model.search(query, device='cuda', threshold=-99, top_k=10)
        val_query_item = dict()
        val_query_item['index'] = [ind for ind, score in results]
        val_query_item['score'] = [score for ind, score in results]
        val_query_result.append(val_query_item)
    json_str = json.dumps(val_query_result, indent=4)
    with open(val_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating val queries result!")

    # test query
    test_query_result = []
    print("Start to create test queries result...")
    for i in tqdm(range(len(test_queries))):
        query = test_queries[i]
        results = model.search(query, device='cuda', threshold=-99, top_k=10)
        test_query_item = dict()
        test_query_item['index'] = [ind for ind, score in results]
        test_query_item['score'] = [score for ind, score in results]
        test_query_result.append(test_query_item)
    json_str = json.dumps(test_query_result, indent=4)
    with open(test_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating test queries result!")


def clean_repetition_datasets(str_data_path, dst_data_path):
    # completed
    str_out_path = str_data_path + "_after_cleaning.txt"
    dst_out_path = dst_data_path + "_after_cleaning.txt"
    documents = []
    dst = []
    with open(str_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            documents.append(line)
    with open(dst_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            dst.append(line)
    print(len(documents))
    print(len(dst))

    new_doc = []
    new_dst = []
    for idx in range(len(documents)):
        if documents[idx] not in new_doc or dst[idx] not in new_dst:
            new_doc.append(documents[idx])
            new_dst.append(dst[idx])
        else:
            print(idx)
            print(documents[idx])
            print(dst[idx])
            print('=' * 30)

    print(len(new_doc))
    print(len(new_dst))
    with open(str_out_path, 'w', encoding='utf-8') as f:
        for doc in new_doc:
            f.write(doc + '\n')
    with open(dst_out_path, 'w', encoding='utf-8') as f:
        for d in new_dst:
            f.write(d + '\n')


def preprocess_wangyue_data(post_path, conv_path, tag_path, src_path, dst_path):
    post_list = []
    with open(post_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            post_list.append(line.strip())

    conv_list = []
    with open(conv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            conv_list.append(line.strip())

    tag_list = []
    with open(tag_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            tag_list.append(line.strip())
    assert len(post_list) == len(conv_list) and len(conv_list) == len(tag_list)

    src_list = []
    dst_list = []
    for i in range(len(post_list)):
        src = post_list[i] + '. ' + conv_list[i]
        dst = tag_list[i].replace(';', ' [SEP] ')
        src_list.append(src)
        dst_list.append(dst)

    with open(src_path, 'w', encoding='utf-8') as f:
        for src in src_list:
            f.write(src + '\n')

    with open(dst_path, 'w', encoding='utf-8') as f:
        for dst in dst_list:
            f.write(dst + '\n')


def compute_hashtag_coverage(labels, hashtags):
    total_r = len(labels)
    total_p = len(hashtags)
    label_list = labels.copy()
    hashtag_list = hashtags.copy()
    true_num = 0
    for lab in label_list:
        for hashtag in hashtag_list:
            if lab == hashtag:
                true_num += 1
                hashtag_list.remove(lab)
                break
    p = true_num / total_p
    r = true_num / total_r
    f = f1(p, r)
    return p, r, f


def hashtag_coverage_cmp(x, y):
    # x is in front of y if the function returns -1
    if x[1][1] + x[1][2] > y[1][1] + y[1][2]:
        return -1
    if x[1][1] + x[1][2] == y[1][1] + y[1][2]:
        if x[1][0] > y[1][0]:
            return -1
        elif x[1][0] == y[1][0]:
            return 0
        else:
            return 1
    return 1


def generate_training_data_for_retrieval(src_data_path, dst_data_path, retrieval_data_path, output_path):
    src, targets, hashtags = get_transformed_io(src_data_path, dst_data_path)
    with open(retrieval_data_path, 'r', encoding='UTF-8') as fp:
        rev_index_list = json.load(fp)
    total_num = len(src)
    positive_samples = []
    hard_negative_samples = []
    for i in tqdm(range(total_num)):
        coverage_rate = dict()
        for j in range(total_num):
            if i == j:
                coverage_rate[i] = (-99, -99, -99)
                continue
            rate = compute_hashtag_coverage(hashtags[i], hashtags[j])
            coverage_rate[j] = rate
        coverage_rate = sorted(coverage_rate.items(), key=cmp_to_key(hashtag_coverage_cmp), reverse=False)[0]
        positive_sample = src[coverage_rate[0]]
        rev_index = rev_index_list[i]["index"]
        nega_rate = dict()
        for ind in rev_index:
            rate = compute_hashtag_coverage(hashtags[i], hashtags[ind])
            nega_rate[ind] = rate
        nega_rate = sorted(nega_rate.items(), key=cmp_to_key(hashtag_coverage_cmp), reverse=True)[:2]
        hard_negative_sample = src[nega_rate[0][0]]
        if hard_negative_sample == positive_sample:
            hard_negative_sample = src[nega_rate[1][0]]
        positive_samples.append(positive_sample)
        hard_negative_samples.append(hard_negative_sample)
    assert len(src) == len(positive_samples) == len(hard_negative_samples)
    with open(output_path, 'w', encoding='UTF-8') as fp:
        header = ['sent0', 'sent1', 'hard_neg']
        writer = csv.writer(fp)
        writer.writerow(header)
        data = []
        for i in range(total_num):
            line = [src[i], positive_samples[i], hard_negative_samples[i]]
            data.append(line)
        writer.writerows(data)


        # print(src[i])
        # print(hashtags[i])
        # print(coverage_rate)
        # for cov in coverage_rate:
        #     print(cov[0])
        #     print(src[cov[0]])
        #     print(hashtags[cov[0]])
        # print('-'*30)


def generate_training_data_for_selector(src_data_path, dst_data_path, output_path):
    src, targets, hashtags = get_transformed_io(src_data_path, dst_data_path)
    constructive_src = []
    positive_samples = []
    hard_negative_samples = []
    for i in tqdm(range(len(src))):
        for hashtag in hashtags[i]:
            hard_negative_hashtag = random_augmentation(hashtag)
            constructive_src.append(src[i])
            positive_samples.append(hashtag)
            hard_negative_samples.append(hard_negative_hashtag)
    total_num = len(constructive_src)
    with open(output_path, 'w', encoding='UTF-8') as fp:
        header = ['sent0', 'sent1', 'hard_neg']
        writer = csv.writer(fp)
        writer.writerow(header)
        data = []
        for i in range(total_num):
            line = [constructive_src[i], positive_samples[i], hard_negative_samples[i]]
            data.append(line)
        writer.writerows(data)


def generate_selector_result_for_retrieval_result(src_data_path, retrieval_index_path, retrieval_document_path, selector_model_path, out_path):
    src_data = []
    with open(src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            src_data.append(line)

    retrieval_document = []
    with open(retrieval_document_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            retrieval_document.append(line)

    with open(retrieval_index_path, 'r', encoding='UTF-8') as fp:
        retrieval_index = json.load(fp)
    assert len(src_data) == len(retrieval_index)
    rev_dst = [[get_hashtag_list(retrieval_document[index]) for index in retrieval_index[i]["index"]] for i in
               range(len(src_data))]
    model = MySimCSE(selector_model_path, device='cuda')
    out_put = []
    for i in range(len(src_data)):
        hashtag_set = []
        for hashtag in rev_dst[i]:
            hashtag_set += hashtag
        hashtag_set = list(set(hashtag_set))
        hashtag_dic = dict()
        model.build_index(hashtag_set, device='cuda', batch_size=64)
        results = model.search(src_data[i], device='cuda', threshold=-99, top_k=99999)
        for ind, score in results:
            hashtag_dic[hashtag_set[ind]] = score
        out_put.append(hashtag_dic)
    json_str = json.dumps(out_put, indent=4, ensure_ascii=False)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def Chinese_delete_stop_words(seq: str):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    words = pseg.cut(seq)
    key_words = [word for word, flag in words if flag not in stop_flag]
    # result = ' '.join(key_words)
    return key_words


def Chinese_bm25_retrieval_score_results(train_src_data_path, val_src_data_path, test_src_data_path):
    train_out_path = train_src_data_path + "_bm25_score.json"
    val_out_path = val_src_data_path + "_bm25_score.json"
    test_out_path = test_src_data_path + "_bm25_score.json"
    # read training documents
    documents = []
    with open(train_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            documents.append(line)
    print("The number of training documents is: ", len(documents))

    # read val queries
    val_queries = []
    with open(val_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            val_queries.append(line)
    print("The number of val queries is: ", len(val_queries))

    # read test queries
    test_queries = []
    with open(test_src_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            test_queries.append(line)
    print("The number of test queries is: ", len(test_queries))

    # build document index
    # split word
    texts = []
    # remove stopwords
    for doc in documents:
        texts.append(Chinese_delete_stop_words(doc))

    bm25_obj = BM25(texts)

    # training query
    train_query_result = []
    print("Start to create training queries result...")
    for i in tqdm(range(len(documents))):
        query = texts[i]
        scores = bm25_obj.get_scores(query)
        best_docs = sorted(range(len(scores)), key=lambda j: scores[j])[-11:][::-1]
        if i in best_docs:
            best_docs.remove(i)
        else:
            best_docs = best_docs[:10]
            print(documents[i])
        train_query_item = dict()
        train_query_item['index'] = best_docs
        train_query_item['score'] = [scores[doc] for doc in best_docs]
        train_query_result.append(train_query_item)
    json_str = json.dumps(train_query_result, indent=4)
    with open(train_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating training queries result!")

    # val query
    val_query_result = []
    print("Start to create val queries result...")
    for i in tqdm(range(len(val_queries))):
        query = Chinese_delete_stop_words(val_queries[i])
        scores = bm25_obj.get_scores(query)
        best_docs = sorted(range(len(scores)), key=lambda j: scores[j])[-10:][::-1]
        val_query_item = dict()
        val_query_item['index'] = best_docs
        val_query_item['score'] = [scores[doc] for doc in best_docs]
        val_query_result.append(val_query_item)
    json_str = json.dumps(val_query_result, indent=4)
    with open(val_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating val queries result!")

    # test query
    test_query_result = []
    print("Start to create test queries result...")
    for i in tqdm(range(len(test_queries))):
        query = Chinese_delete_stop_words(test_queries[i])
        scores = bm25_obj.get_scores(query)
        best_docs = sorted(range(len(scores)), key=lambda j: scores[j])[-10:][::-1]
        test_query_item = dict()
        test_query_item['index'] = best_docs
        test_query_item['score'] = [scores[doc] for doc in best_docs]
        test_query_result.append(test_query_item)
    json_str = json.dumps(test_query_result, indent=4)
    with open(test_out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("Finish creating test queries result!")


if __name__ == '__main__':
    src_train_data_path = 'data/THG_twitter/twitter.2021.train.src_after_cleaning.txt'
    dst_data_path = 'data/THG_twitter/twitter.2021.train.dst_after_cleaning.txt'
    src_val_data_path = 'data/THG_twitter/twitter.2021.valid.src_after_cleaning.txt'
    src_test_data_path = 'data/THG_twitter/twitter.2021.test.src_after_cleaning.txt'
    rev_index_path = 'data/THG_twitter/twitter.2021.train.src_after_cleaning.txt_simcse_tuned_dense_score.json'

    # compute_hashtag_coverage(['fanconi anemia', '12345'], ['anemia', '9977'])
    # retrieval_training_data_path = 'data/THG_twitter/retrieval_training_data.csv'
    # generate_training_data_for_retrieval(src_train_data_path, dst_data_path, rev_index_path, retrieval_training_data_path)

    # selector_training_data_path = 'data/THG_twitter/selector_training_data.csv'
    # generate_training_data_for_selector(src_train_data_path, dst_data_path, selector_training_data_path)

    # generate_selector_result_for_retrieval_result(src_train_data_path, rev_index_path, dst_data_path, './result/selector-tuned-sup-simcse-roberta-large')

    # post_path = 'data/Twitter_naacl_wangyue/test_post.txt'
    # conv_path = 'data/Twitter_naacl_wangyue/test_conv.txt'
    # tag_path = 'data/Twitter_naacl_wangyue/test_tag.txt'
    # src_path = 'data/Twitter_naacl_wangyue/test_src.txt'
    # dst_path = 'data/Twitter_naacl_wangyue/test_dst.txt'
    # preprocess_wangyue_data(post_path, conv_path, tag_path, src_path, dst_path)

    # wangyue_train_src_path = 'data/Twitter_naacl_wangyue/train_src.txt'
    # wangyue_val_src_path = 'data/Twitter_naacl_wangyue/valid_src.txt'
    # wangyue_test_src_path = 'data/Twitter_naacl_wangyue/test_src.txt'

    # src_train_data_path = 'data/THG_twitter/sample_src.txt'
    # src_val_data_path = 'data/THG_twitter/sample_src.txt'
    # src_test_data_path = 'data/THG_twitter/sample_src.txt'
    # generate_index_json_file(data_path)
    # dense_retrieval_results(data_path)

    # src_data_path = 'data/WHG/new4_test.src'
    # dst_data_path = 'data/WHG/new4_test.dst'
    # clean_repetition_datasets(src_data_path, dst_data_path)

    # dense_retrieval_results(src_train_data_path, src_val_data_path, src_test_data_path)
    # bm25_retrieval_score_results(wangyue_train_src_path, wangyue_val_src_path, wangyue_test_src_path)

    train_src_data_path = 'data/WHG/new4_train.src_after_cleaning.txt'
    val_src_data_path = 'data/WHG/new4_validation.src'
    test_src_data_path = 'data/WHG/new4_test.src'
    # Chinese_bm25_retrieval_score_results(train_src_data_path, val_src_data_path, test_src_data_path)

    dense_retrieval_results(train_src_data_path, val_src_data_path, test_src_data_path)

    # src_train_data_path = 'data/WHG/new4_train.src_after_cleaning.txt'
    # dst_data_path = 'data/WHG/new4_train.dst_after_cleaning.txt'
    # src_val_data_path = 'data/WHG/new4_validation.src'
    # src_test_data_path = 'data/WHG/new4_test.src'
    # rev_index_train_path = 'data/WHG/new4_train.src_after_cleaning.txt_bert_dense_score.json'
    # rev_index_val_path = 'data/WHG/new4_validation.src_bert_dense_score.json'
    # rev_index_test_path = 'data/WHG/new4_test.src_bert_dense_score.json'
    # out_train_path = 'data/WHG/train_tunedbert_selector_result.json'
    # out_val_path = 'data/WHG/validation_tunedbert_selector_result.json'
    # out_test_path = 'data/WHG/test_tunedbert_selector_result.json'
    # generate_selector_result_for_retrieval_result(src_val_data_path, rev_index_val_path, dst_data_path, '/home/qiupeng/frz_project/SimCSE/result/selector_bert_chinese_base', out_val_path)
    # generate_selector_result_for_retrieval_result(src_test_data_path, rev_index_test_path, dst_data_path, '/home/qiupeng/frz_project/SimCSE/result/selector_bert_chinese_base', out_test_path)
    # generate_selector_result_for_retrieval_result(src_train_data_path, rev_index_train_path, dst_data_path, '/home/qiupeng/frz_project/SimCSE/result/selector_bert_chinese_base', out_train_path)
