import sys
import re
import json

import requests
from django.conf import settings
import jpype
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
import numpy as np


def main(args, headline, body):
    jpype.attachThreadToJVM()
    print("1. Dataset split")
    if True:
        print("2. Morpheme analysis")
        headline = morpheme_analysis_konlpy(headline)
        body = convert_body(body)
        #body = prettify_body(body)
        paras = body.split("<EOP>")
        paras_processed = []
        for para in paras:
            if len(para) == 0:
                continue
            para = morpheme_analysis_konlpy(para)
            paras_processed.append(para)
        print(headline)
        
        for para in paras:
            print(para + "\n")

        return get_score(args, headline, paras_processed)


def prettify_body(body_text):

    regex_brackets = r"\(.*?\)|\[.*?\]|\{.*?\}|\<.*?\>"
    body_text = re.sub(regex_brackets, "", body_text)

    regex_email = r"[^\s@]+@[^\s@]+\.[^\s@]+"
    body_text = re.sub(regex_email, "", body_text)

    regex_mutliple_n = r'[\n]{1,}'
    body_text = re.sub(regex_mutliple_n, '\n', body_text)


    paragraphs = body_text.split("\n")


    paragraphs = [para.strip() for para in paragraphs]
    paragraphs = filter(None, paragraphs)
    
    final_paragraphs = []
    paragraph_without_period = []
    for paragraph in paragraphs:

        if paragraph[-1] not in [".", "?", ","]:
            continue
        
        sentences = sent_tokenize(paragraph)
        morphemed_sentences = []
        for sentence in sentences:
            sentence = " ".join(okt.morphs(sentence))
            morphemed_sentences.append(sentence)

        paragraph = " <EOS> ".join(morphemed_sentences)
        
        final_paragraphs.append(paragraph)

    final_paragraph = " <EOP> ".join(final_paragraphs)

    return final_paragraph

def convert_body(body_text):

    _body = body_text
        # 문단 구분
    __body = _body.replace('\n',"<EOP>")
    #__body = __body.replace('\n', "")


    # 2개이상 연속적인 공백 제거
    regex = '(\s{2,})'
    __paragraph_list = []
    res = re.finditer(regex, __body)
    prev_eos = 0
    for r in res:
        curr_eos = r.end()
        sent = __body[prev_eos:curr_eos].strip()
        if sent != ' ' and sent:
            __paragraph_list.append(sent)
        prev_eos = curr_eos
    if prev_eos < len(__body):
        sent = __body[prev_eos:].strip()
        if sent != ' ' and sent:
            __paragraph_list.append(sent)
    ___body = ''.join(__paragraph_list)


    # 문장 구분
    para_list = []
    for para in ___body.split("<EOP>"):
        ____body = split_sentence(para)
        if ____body is not None:
            para_list.append(____body)
    pre_p_body = "<EOP>".join(para_list) + "<EOP>"

    pre_p_body = morpheme_analysis_konlpy(pre_p_body)

    pre_p_body = pre_p_body.replace("< EOP >", "<EOP> ")
    pre_p_body = pre_p_body.replace("< EOS >", "")

    return pre_p_body

def split_sentence(paragraph):
    # print("paragraph:", paragraph)
    # paragraph = "예시 문장입니다. "
    # paragraph += "이런것도 문장으로 처리 함 (두번 째 정규식에 포함). "
    # paragraph += "마지막 문장은 점이 없어도 빠짐덦이 처리합니다"

    # print(paragraph)

    # regex = '[가-힣]+₩.|[가-힣]+₩s?₩)₩.'
    # regex = '[가-힣]+\.|[가-힣]+\s?\)\.|[가-힣]+\?|[가-힣]+\,'
    regex = '[가-힣]+\.|[가-힣]+\s?\)\.|[가-힣]+\?'

    sentence_list = []

    res = re.finditer(regex, paragraph)
    prev_eos = 0
    # print("token")
    for r in res:
        curr_eos = r.end()
        sent = paragraph[prev_eos:curr_eos].strip()
        if sent != ' ' and sent:
            sentence_list.append(sent)
        prev_eos = curr_eos
        # print(sent)
    if prev_eos < len(paragraph):
        sent = paragraph[prev_eos:].strip()
        if sent != ' ' and sent:
            sentence_list.append(sent)

    if len(sentence_list) == 0:
        return None
    else:
        return " <EOS> ".join(sentence_list) + " <EOS> "


def morpheme_analysis_konlpy(text):
    text = " ".join(settings.MORPHEME_MODEL.morphs(text))
    return text


def test_mission_1(args):
    args.max_headline_len = 50
    args.max_body_len = 500

    early_stopping = EarlyStopping(patience=args.num_patience)
    history = CustomHistory()
    history.init()

    args.id_dir = args.train_id_dir
    args.headline_dir = args.train_headline_dir
    args.sentence_dir = args.train_sentence_dir

    (x_h_train, x_h_valid, x_h_test,
     x_b_train, x_b_valid, x_b_test,
     y_train, y_valid, y_test) \
        = pp.split_dataset_for_train_word(args)


    x_h_train = sequence.pad_sequences(x_h_train, maxlen=args.max_headline_len, dtype='float32')
    x_h_valid = sequence.pad_sequences(x_h_valid, maxlen=args.max_headline_len, dtype='float32')
    x_h_test = sequence.pad_sequences(x_h_test, maxlen=args.max_headline_len, dtype='float32')
    x_b_train = sequence.pad_sequences(x_b_train, maxlen=args.max_body_len, dtype='float32')
    x_b_valid = sequence.pad_sequences(x_b_valid, maxlen=args.max_body_len, dtype='float32')
    x_b_test = sequence.pad_sequences(x_b_test, maxlen=args.max_body_len, dtype='float32')


    print(x_h_train.shape)
    print(x_h_valid.shape)
    print(x_h_test.shape)
    print(x_b_train.shape)
    print(x_b_valid.shape)
    print(x_b_test.shape)


    model = model_selector(args, None)

    print('Train...')
    model.fit([x_h_train, x_b_train], y_train,
              epochs=args.num_epochs, batch_size=args.batch_size,
              validation_data=([x_h_valid, x_b_valid], y_valid),
              shuffle=True, callbacks=[history, early_stopping])
    # model.fit([x_train[:, :args.max_headline_len], x_train[:, args.max_headline_len:]], y_train,
    #           epochs=args.num_epochs, batch_size=args.batch_size,
    #           validation_data = ([x_valid[:, :args.max_headline_len], x_valid[:, args.max_headline_len:]], y_valid),
    #           shuffle=True, callbacks=[history, early_stopping])


    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(history.train_loss, 'y', label='train loss')
    loss_ax.plot(history.val_loss, 'r', label='val loss')

    acc_ax.plot(history.train_acc, 'b', label='train acc')
    acc_ax.plot(history.val_acc, 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')


    print("Evaluate...")
    # evaluate model
    scores = model.evaluate([x_h_test, x_b_test], y_test, batch_size=args.batch_size)
    # scores = model.evaluate(x_test, y_test, batch_size=args.batch_size)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    model.save(args.data_dir + args.mission1_model_dir)




    print("test")
    args.id_dir = args.dest_dir + "id.csv"
    args.headline_dir = args.morpheme_dest_dir + "head_mor.csv"
    args.body_dir = args.morpheme_dest_dir + "body_mor.csv"

    (array_article, x_h_test, x_b_test, y_test) \
        = pp.split_dataset_for_test_word(args)

    x_h_test = sequence.pad_sequences(x_h_test, maxlen=args.max_headline_len, dtype='float32')
    x_b_test = sequence.pad_sequences(x_b_test, maxlen=args.max_body_len, dtype='float32')
    yhat = model.predict([x_h_test, x_b_test], batch_size=args.batch_size)


    with open(args.data_dir + args.output_dir + args.mission + ".txt", "w") as f:
        for idx, y in enumerate(yhat):
            f.write("%s,%0.7f\n" % (array_article[idx][0], y))

    # with open(args.data_dir + args.output_dir + args.mission + ".txt", "w") as f:
    #     for idx, y in enumerate(yhat):
    #         f.write("%s,%0.7f,%s\n" % (array_article[idx][0], y, array_article[idx][2]))

    # scores = model.evaluate([x_h_test, x_b_test], y_test, batch_size=args.batch_size)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def get_score(args, headline, paras):

    print("3. Data preprocessing")

    vocab = settings.WORD2IDX

    zero_pads = np.zeros(80, dtype=np.int32).tolist()

    headline = headline.split(" ")
    translated_headline = []
    for word in headline:
        if word in vocab:
            translated_headline.append(vocab[word])

    headline_seq_length = len(translated_headline)
    headline_seq_length = 49 if headline_seq_length > 49 else headline_seq_length

    translated_paras = []
    paras_seq_length = []
    for i, paras in enumerate(paras):
        word_list = paras.split(" ")
        translated_word_list = []
        for word in word_list:
            if word in vocab:
                translated_word_list.append(vocab[word])
       
        if len(paras_seq_length) < 10:
            para_len = len(translated_word_list)
            para_len = 80 if para_len > 80 else para_len
            paras_seq_length.append(para_len)
        
        translated_paras.append((translated_word_list + zero_pads)[:80])

    paras_seq_length = (paras_seq_length + zero_pads)[:10]
    context_seq_length = len(translated_paras)
    context_seq_length = 10 if context_seq_length > 10 else context_seq_length
    
    translated_paras = (translated_paras + ([zero_pads] * 10))[:10]

    translated_headline = (translated_headline + zero_pads)[:49]
    translated_headline = [translated_headline]


    post_data = {
        "encoder_input": translated_paras,
        "encoderR_input": translated_headline,
        "encoder_seq_length": paras_seq_length,
        "context_seq_length": [context_seq_length],
        "encoderR_seq_length": [headline_seq_length],
        "dr_text_in": 1.0,
        "dr_text_out_ph": 1.0,
        "dr_con_in_ph": 1.0,
        "dr_con_out_ph": 1.0,
        "label": [[1]]
        }

    for key in post_data:
        print(key)
        print(post_data[key])

    headers = {"content-type": "application/json"}
    r = requests.post('http://localhost:8501/v1/models/saved_model:predict', data=json.dumps({"inputs": post_data}), headers=headers)
    print(r.json())

    return r.json()["outputs"][0][0]


if __name__ == "__main__":
    pass
