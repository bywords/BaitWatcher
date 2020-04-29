from __future__ import division
from __future__ import print_function
import csv, json
import numpy as np



def load_mission1_data(args):
    return _readID(args), _readHeadline(args), _readBody(args)

def load_mission2_data(args):
    return _readID(args), _readHeadline(args), _readParagraphID(args), _readParagraph(args)

def load_mission2_data_for_train_sentence(args):
    return _readID(args), _readHeadline(args), _readSentence(args)

def load_mission2_data_for_test_sentence(args):
    return _readID(args), _readHeadline(args), _readSentence(args)


# 헤드라인 및 본문 내용의 아이디 정보 읽기 - return type: list
def _readID(args):
    arr = []
    fname = args.data_dir + args.id_dir
    print("\tload %s" % fname)
    with open(fname, newline='', encoding='cp949', errors='ignore') as file:
        arr = list(csv.reader(file))
        arr.pop(0)
    return arr


# 해드라인 읽기 - return type: dictionary(id, text)
def _readHeadline(args):
    dic_head = {}
    fname = args.data_dir + args.headline_dir
    print("\tload %s" % fname)
    with open(fname, newline='', encoding='cp949', errors='ignore') as file:
        for row in csv.reader(file):
            dic_head[row[0]] = row[1]
    return dic_head


# 본문 읽기 - return type: dictionary(id, text)
def _readBody(args):
    dic_body = {}
    fname = args.data_dir + args.body_dir
    print("\tload %s" % fname)
    with open(fname, newline='', encoding='cp949', errors='ignore') as file:
        for row in csv.reader(file):
            dic_body[row[0]] = row[1]
    return dic_body



# 본문 - 단락 단위로 읽기 - return type: dictionary(id, list)
def _readParagraphID(args):
    dic_paragraph_id = {}
    fname = args.data_dir + args.paragraph_id_dir
    print("\tload %s" % fname)
    i = 0
    with open(fname, newline='', encoding='utf-8', errors='ignore') as file:
        for row in csv.reader(file):
            # print(row)
            i = i + 1
            dic_paragraph_id[row[0]] = row[2]
            # if i == 10:
            #     break
    return dic_paragraph_id


# 단락 단위로 읽기 - return type: dictionary(id, text)
def _readParagraph(args):
    dic_paragraph = {}
    fname = args.data_dir + args.paragraph_dir
    print("\tload %s" % fname)
    i = 0
    with open(fname, newline='', encoding='cp949', errors='ignore') as file:
        for row in csv.reader(file):
            i = i + 1
            # print(row)
            dic_paragraph[row[0]] = row[1]
            # if i == 3000:
            #     break
    return dic_paragraph


def _readSentence(args):
    import sys
    csv.field_size_limit(sys.maxsize)
    dic_sentence = {}
    fname = args.data_dir + args.sentence_dir
    print("\tload %s" % fname)
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences = json.loads(row['text'], encoding='cp949')
            dic_sentence[row['id']] = sentences
    return dic_sentence



"""
Load word -> index and index -> word mappings
:param vocab_path: where the word-index map is saved
:return: word2idx, idx2word
"""
def load_vocab(args):
    fname = args.data_dir + args.vocab_file_path
    print("\tload %s" % fname)
    with open(fname, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word



"""
Generate an embedding layer word embeddings
:param embeddings_path: where the embeddings are saved (as a numpy file)
:return: the generated embedding layer
"""
def load_embedding_matrix(args):
    embeddings_path = args.data_dir + args.embedding_file_path
    weights = np.load(open(embeddings_path, 'rb'))
    args.nb_words = weights.shape[0]
    args.embedding_dim = weights.shape[1]
    print("\tload embedding matrix using word2vec model: (%d, %d): %s" % (args.nb_words, args.embedding_dim, embeddings_path))
    return weights


