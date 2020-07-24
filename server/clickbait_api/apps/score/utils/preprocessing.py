#!/usr/bin/python3
import subprocess
import csv
import os
import sys
import numpy as np
from apps.score.utils import filereader as fr
from gensim.utils import simple_preprocess
from django.conf import settings
import jpype
from pprint import pprint

# fix the random seed
np.random.seed(400)

# Done for the Error
csv.field_size_limit(sys.maxsize)


def split_dataset_for_train_word(args):
    # tokenizer: can change this as needed
    tokenize = lambda x: simple_preprocess(x)

    # 데이터 읽기
    array_article, dic_headline, dic_sentence = fr.load_mission2_data_for_train_sentence(args)
    word2idx, idx2word = fr.load_vocab(args)
    weight = fr.load_embedding_matrix(args)

    x_h_arr = []
    x_b_arr = []
    y_arr = []
    i = 0

    import random
    SEED = 400
    random.seed(SEED)
    random.shuffle(array_article)

    # array_article = array_article[:100]
    for dic in array_article:
        i = i + 1
        timestep_h_array = []
        timestep_b_array = []
        head = dic_headline[dic[0]]
        sentence_list = dic_sentence[dic[1]]
        label = dic[2]

        # Headline
        word_list = tokenize(head)
        timestep_h_array = _custom_embedding_2(word_list, word2idx, weight)

        # Paragraph
        for sentence in sentence_list:
            word_list = tokenize(sentence)
            timestep_b_array = timestep_b_array + _custom_embedding_2(word_list, word2idx, weight)
            # sentence_array = _custom_embedding_2(word_list, word2idx, weight)
            # if sentence_array is not None:
            #     timestep_b_array.append(sentence_array)


        # print(len(timestep_h_array))

        x_h_arr.append(timestep_h_array)
        x_b_arr.append(timestep_b_array)
        y_arr.append(int(label))

        if i % 10000 == 0:
            print("Processing input text - ", i)

    trian_size =  int(len(x_h_arr) * args.train_split)
    valid_size = int(len(x_h_arr) * args.validation_split)

    x_h_train = x_h_arr[:trian_size]
    x_h_valid = x_h_arr[trian_size:trian_size+valid_size]
    x_h_test = x_h_arr[trian_size+valid_size:]
    x_b_train = x_b_arr[:trian_size]
    x_b_valid = x_b_arr[trian_size:trian_size+valid_size]
    x_b_test = x_b_arr[trian_size+valid_size:]
    y_train = y_arr[:trian_size]
    y_valid = y_arr[trian_size:trian_size + valid_size]
    y_test = y_arr[trian_size + valid_size:]

    return (x_h_train, x_h_valid, x_h_test, x_b_train, x_b_valid, x_b_test, y_train, y_valid, y_test)



def split_dataset_for_test_word(args):
    # tokenizer: can change this as needed
    tokenize = lambda x: simple_preprocess(x)

    # 데이터 읽기
    array_article, dic_headline, dic_body = fr.load_mission1_data(args)
    word2idx, idx2word = fr.load_vocab(args)
    weight = fr.load_embedding_matrix(args)

    x_h_arr = []
    x_b_arr = []
    y_arr = []
    i = 0

    # shuffle(array_article)
    # array_article = array_article[:500000]
    for dic in array_article:
        i = i + 1
        timestep_h_array = []
        timestep_b_array = []
        head = dic_headline[dic[0]]
        body = dic_body[dic[1]]
        label = dic[2]

        # Headline
        word_list = tokenize(head)
        timestep_h_array = _custom_embedding_2(word_list, word2idx, weight)

        word_list = tokenize(body)
        timestep_b_array = _custom_embedding_2(word_list, word2idx, weight)

        x_h_arr.append(timestep_h_array)
        x_b_arr.append(timestep_b_array)
        y_arr.append(int(label))

        if i % 10000 == 0:
            print("Processing input text - ", i)

    trian_size =  int(len(x_h_arr) * args.train_split)
    valid_size = int(len(x_h_arr) * args.validation_split)

    x_h_test = x_h_arr
    x_b_test = x_b_arr
    y_test = y_arr

    return (array_article, x_h_test, x_b_test, y_test)





def split_dataset_for_test_paragraph(args):
    # tokenizer: can change this as needed
    tokenize = lambda x: simple_preprocess(x)

    # 데이터 읽기
    array_article, dic_headline, dic_paragraph_id, dic_paragraph = fr.load_mission2_data(args)
    print("4. Loading embedding matrix")
    word2idx, idx2word = fr.load_vocab(args)
    weight = fr.load_embedding_matrix(args)

    x_h_arr = []
    x_b_arr = []
    y_arr = []
    i = 0

    # shuffle(array_article)
    # array_article = array_article[:12000]
    for dic in array_article:
        i = i + 1
        timestep_h_array = []
        timestep_b_array = []
        head = dic_headline[dic[0]]
        paragraph_list = dic_paragraph_id[dic[1]].split(",")
        label = dic[2]

        # Headline
        word_list = tokenize(head)
        head_array = _custom_embedding(word_list, word2idx, weight)
        if head_array is not None:
            timestep_h_array.append(head_array)

        # Paragraph
        for id in paragraph_list:
            word_list = tokenize(dic_paragraph[id])
            paragraph_array = _custom_embedding(word_list, word2idx, weight)
            if paragraph_array is not None:
                timestep_b_array.append(paragraph_array)


        x_h_arr.append(timestep_h_array)
        x_b_arr.append(timestep_b_array)
        y_arr.append(int(label))

        # if i % 100 == 0:
        #     print("Processing input text - ", i)

    x_h_test = x_h_arr
    x_b_test = x_b_arr
    y_test = y_arr
    return (array_article, x_h_test, x_b_test, y_test)



def split_dataset_for_test_paragraph_2(args):
    # tokenizer: can change this as needed
    tokenize = lambda x: simple_preprocess(x)

    # 데이터 읽기
    array_article, dic_headline, dic_paragraph_id, dic_paragraph = fr.load_mission2_data(args)
    word2idx, idx2word = fr.load_vocab(args)
    weight = fr.load_embedding_matrix(args)

    x_arr = []
    y_arr = []
    i = 0

    # shuffle(array_article)
    # array_article = array_article[:12000]
    for dic in array_article:
        i = i + 1
        timestep_array = []
        """try:
            
        except KeyError:
            print(dic[0])
            print(dic_headline)"""
        head = dic_headline[dic[0]]
        print("head : {}".format(head))
        paragraph_list = dic_paragraph_id[dic[1]].split(",")
        #print("pa_list : {}".format(paragraph_list))
        label = dic[2]
        #print("labal : {}".format(label))

        # Headline
        word_list = tokenize(head)
        head_array = _custom_embedding(word_list, word2idx, weight)
        if head_array is not None:
            timestep_array.append(head_array)

        # Paragraph
        for id in paragraph_list:
            word_list = tokenize(dic_paragraph[id])
            paragraph_array = _custom_embedding(word_list, word2idx, weight)
            if paragraph_array is not None:
                timestep_array.append(paragraph_array)


        x_arr.append(timestep_array)
        y_arr.append(int(label))

        # if i % 100 == 0:
        #     print("Processing input text - ", i)

    x_test = x_arr
    y_test = y_arr
    return (array_article, x_test, y_test)


def translate_text_to_index(args, headline, paras):
    # tokenizer: can change this as needed
    tokenize = lambda x: simple_preprocess(x)

    # 데이터 읽기
    array_article, dic_headline, dic_paragraph_id, dic_paragraph = fr.load_mission2_data(args)
    word2idx, idx2word = fr.load_vocab(args)
    weight = fr.load_embedding_matrix(args)

    x_arr = []
    y_arr = []
    i = 0

    # shuffle(array_article)
    # array_article = array_article[:12000]
    for dic in array_article:
        i = i + 1
        timestep_array = []
        """try:
            
        except KeyError:
            print(dic[0])
            print(dic_headline)"""
        head = dic_headline[dic[0]]
        print("head : {}".format(head))
        paragraph_list = dic_paragraph_id[dic[1]].split(",")
        #print("pa_list : {}".format(paragraph_list))
        label = dic[2]
        #print("labal : {}".format(label))

        # Headline
        word_list = tokenize(head)
        head_array = _custom_embedding(word_list, word2idx, weight)
        if head_array is not None:
            timestep_array.append(head_array)

        # Paragraph
        for id in paragraph_list:
            word_list = tokenize(dic_paragraph[id])
            paragraph_array = _custom_embedding(word_list, word2idx, weight)
            if paragraph_array is not None:
                timestep_array.append(paragraph_array)


        x_arr.append(timestep_array)
        y_arr.append(int(label))

        # if i % 100 == 0:
        #     print("Processing input text - ", i)

    x_test = x_arr
    y_test = y_arr
    return (array_article, x_test, y_test)




def _custom_embedding(array, word2idx, weight):
    for idx, word in enumerate(array):
        try:
            val = word2idx[word]
        except KeyError:
            array[idx] = -1
            pass
        else:
            array[idx] = val
    array = list(filter((-1).__ne__, array))
    for idx, word_idx in enumerate(array):
        try:
            val = weight[word_idx]
        except KeyError:
            print(word_idx)
            pass
        else:
            array[idx] = val
    if len(array) == 0:
        return None
    return np.asarray(np.mean(array, axis=0))
    # return array


def _custom_embedding_2(array, word2idx, weight):
    for idx, word in enumerate(array):
        try:
            val = word2idx[word]
        except KeyError:
            array[idx] = -1
            pass
        else:
            array[idx] = val
    array = list(filter((-1).__ne__, array))
    for idx, word_idx in enumerate(array):
        try:
            val = weight[word_idx]
        except KeyError:
            print(word_idx)
            pass
        else:
            array[idx] = val
    
    return array




"""형태소 분석"""
def morpheme_analysis(morpheme_dir, source_dir, dest_dir):
    cmd = "%s %s %s" % (morpheme_dir, source_dir, dest_dir)
    print(cmd)
    return subprocess.call(cmd, shell=True)

def morpheme_analysis_konlpy(morpheme_dir, source_dir, dest_dir):
    jpype.attachThreadToJVM()
    source_file = open(source_dir, encoding="utf-8-sig")
    source_file.readline()
    dest_file = open(dest_dir, "w", encoding="cp949")
    csvReader = csv.reader(source_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    csvWriter = csv.writer(dest_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in csvReader:
        if len(row) != 2:
            raise AttributeError
        
        id = row[0]
        text = row[1]
        text = " ".join(settings.MORPHEME_MODEL.morphs(text))
        csvWriter.writerow([id, text])
    source_file.close()
    dest_file.close()
    return 1






"""
format_raw_csv(source_file_dir, dest_dir, source_file_encoding = "utf_16_le")
    - 임무 파일을 id, body, head 로 분할합니다.
    - params
        - source_file_dir : 원본 임무 파일이 위치한 경로 (ex. ./test/임무1_모의테스트_V1.csv)
        - dest_dir : id, head, body로 분리된 파일이 저장될 폴더의 경로. (ex. ./test_result/)
        - source_file_encoding : 원본 임무 파일의 인코딩 형식. 모의 테스트의 인코딩이었던 utf_16_le 가 기본값.

format_raw_csv_paragraph(source_file_dir, dest_dir, source_file_encoding = "utf_16_le")
    - 임무 파일을 id, body, head, paragraph으로 분할합니다.
    - 파라미터는 위와 동일합니다.
"""


def format_raw_csv(source_file_dir, dest_dir, source_file_encoding = "utf_16_le"):
    baseCSV = open(source_file_dir, "r", encoding=source_file_encoding, newline="")
    baseCSV.readline()
    csvReader = csv.reader(baseCSV, delimiter="\t")
    headCSV = open(os.path.join(dest_dir, "head.csv"), "w", encoding="utf-8-sig")
    bodyCSV = open(os.path.join(dest_dir, "body.csv"), "w", encoding="utf-8-sig")
    idCSV = open(os.path.join(dest_dir, "id.csv"), "w", encoding="utf-8-sig")
    headHeader = ['headline_id', 'text']
    bodyHeader = ['article_id', 'text']
    idHeader = ['headline_id', 'article_id', 'label', 'difficulty']
    headWriter = csv.DictWriter(headCSV, fieldnames=headHeader)
    bodyWriter = csv.DictWriter(bodyCSV, fieldnames=bodyHeader)
    idWriter = csv.DictWriter(idCSV, fieldnames=idHeader)
    headWriter.writeheader()
    bodyWriter.writeheader()
    idWriter.writeheader()

    for row in csvReader:
        row_id = row[0]
        row_head = row[1]
        row_body = row[2]
        headWriter.writerow({
            'headline_id': row_id,
            'text': row_head
            })
        bodyWriter.writerow({
            'article_id': row_id,
            'text': row_body
            })
        idWriter.writerow({
            'headline_id': row_id,
            'article_id': row_id,
            'label': 0,
            'difficulty': 0
            })

    headCSV.close()
    bodyCSV.close()
    idCSV.close()
    baseCSV.close()


def format_raw_csv_paragraph(source_file_dir, dest_dir, source_file_encoding = "utf-8-sig", delimiter=","):
    #print(source_file_dir)
    baseCSV = open(source_file_dir, "r", encoding=source_file_encoding, newline="")
    baseCSV.readline()
    csvReader = csv.reader(baseCSV, delimiter=delimiter)
    headCSV = open(os.path.join(dest_dir, "head.csv"), "w", encoding="utf-8-sig")
    bodyCSV = open(os.path.join(dest_dir, "body.csv"), "w", encoding="utf-8-sig")
    idCSV = open(os.path.join(dest_dir, "id.csv"), "w", encoding="utf-8-sig")
    pgCSV = open(os.path.join(dest_dir, "paragraph.csv"), "w", encoding="utf-8-sig")
    headHeader = ['headline_id', 'text']
    bodyHeader = ['article_id', 'text', 'paragraph_ids']
    idHeader = ['headline_id', 'article_id', 'label', 'difficulty']
    pgHeader = ['paragraph_id', 'text']
    headWriter = csv.DictWriter(headCSV, fieldnames=headHeader)
    bodyWriter = csv.DictWriter(bodyCSV, fieldnames=bodyHeader)
    idWriter = csv.DictWriter(idCSV, fieldnames=idHeader)
    pgWriter = csv.DictWriter(pgCSV, fieldnames=pgHeader)
    headWriter.writeheader()
    bodyWriter.writeheader()
    idWriter.writeheader()
    pgWriter.writeheader()

    pg_count = 0

    for row in csvReader:
        row_id = row[0]
        row_head = row[1]
        row_body = row[2]
        try:
            row_label = row[3]
        except IndexError:
            row_label = 0
            pass
        para_ids = []

        # BODY to PARAGRAPH
        paras_raw = row_body.split("\n")
        for para in paras_raw:
            if len(para) == 0:
                continue
            pg_id = "1{0:05d}".format(pg_count)
            pg_count += 1
            pgWriter.writerow({
                'paragraph_id': pg_id,
                'text': para
                })
            para_ids.append(pg_id)
        headWriter.writerow({
            'headline_id': row_id,
            'text': row_head
            })
        bodyWriter.writerow({
            'article_id': row_id,
            'text': row_body,
            'paragraph_ids': ",".join(para_ids)
            })
        idWriter.writerow({
            'headline_id': row_id,
            'article_id': row_id,
            'label': row_label,
            'difficulty': 0
            })

    headCSV.close()
    bodyCSV.close()
    idCSV.close()
    baseCSV.close()