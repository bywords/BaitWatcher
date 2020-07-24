import sys

from keras.preprocessing import sequence
from keras.models import load_model
from keras.callbacks import EarlyStopping

import utils.preprocessing as pp
from utils import argumentparser
from utils.history import CustomHistory
from model.model_2 import model_selector



def main(args):
    source = args.data_dir + args.source_dir + args.dataset
    dest = args.data_dir+args.dest_dir

    print("1. Dataset split")
    # pp.format_raw_csv_paragraph(source, dest, source_file_encoding="utf-8", delimiter=",")
    pp.format_raw_csv_paragraph(source, dest)
    print("\tload source dataset:", source)
    print("\tsave split dataset:", dest)



    


    if args.mission == "mission1":


        print("2. Morpheme analysis")
        morpheme_dir = args.data_dir + args.morpheme_dir
        mor_source_head = args.data_dir + args.dest_dir + "head.csv"
        mor_source_body = args.data_dir + args.dest_dir + "body.csv"
        mor_dest_head = args.data_dir + args.morpheme_dest_dir + "head_mor.csv"
        mor_dest_body = args.data_dir + args.morpheme_dest_dir + "body_mor.csv"
        result = pp.morpheme_analysis(morpheme_dir, mor_source_head, mor_dest_head) # Headline
        if result == "1":
            print("\tload morpheme source :", mor_source_head)
            print("\tsave morpheme dest:", mor_dest_head)

        result = pp.morpheme_analysis(morpheme_dir, mor_source_body, mor_dest_body) # Headline
        if result == "1":
            print("\tload morpheme source :", mor_source_body)
            print("\tsave morpheme dest:", mor_dest_body)


        test_mission_1(args)

    elif args.mission == "mission2":


        print("2. Morpheme analysis")
        morpheme_dir = args.data_dir + args.morpheme_dir
        mor_source_head = args.data_dir + args.dest_dir + "head.csv"
        mor_source_para = args.data_dir + args.dest_dir + "paragraph.csv"
        mor_dest_head = args.data_dir + args.morpheme_dest_dir + "head_mor.csv"
        mor_dest_para = args.data_dir + args.morpheme_dest_dir + "paragraph_mor.csv"
        result = pp.morpheme_analysis(morpheme_dir, mor_source_head, mor_dest_head) # Headline
        if result == "1":
            print("\tload morpheme source :", mor_source_head)
            print("\tsave morpheme dest:", mor_dest_head)

        result = pp.morpheme_analysis(morpheme_dir, mor_source_para, mor_dest_para) # Paragraph
        if result == "1":
            print("\tload morpheme source :", mor_source_para)
            print("\tsave morpheme dest:", mor_dest_para)


        args.id_dir = args.dest_dir + "id.csv"
        args.headline_dir = args.morpheme_dest_dir + "head_mor.csv"
        args.paragraph_id_dir = args.dest_dir + "body.csv"
        args.paragraph_dir = args.morpheme_dest_dir + "paragraph_mor.csv"
        test_mission_2(args)

    else:
        print("Error: mission type")







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






def test_mission_2(args):
    args.max_sequence_len = 30

    print("3. Data preprocessing")
    (array_article, x_test, y_test) = pp.split_dataset_for_test_paragraph_2(args)

    

    print("5. Loading model")
    model = load_model(args.data_dir + args.mission2_model_dir)
    print("\tload", args.data_dir + args.mission2_model_dir)

    x_test = sequence.pad_sequences(x_test, maxlen=args.max_sequence_len, dtype='float32')

    print("6. Prediction")
    yhat = model.predict(x_test, batch_size=args.batch_size)

    with open(args.data_dir + args.output_dir + args.mission + ".txt", "w") as f:
        for idx, y in enumerate(yhat):
            f.write("%s,%0.7f\n" % (array_article[idx][0], y))


    # with open(args.data_dir + args.output_dir + args.mission + ".txt", "w") as f:
    #     for idx, y in enumerate(yhat):
    #         f.write("%s,%0.7f,%s\n" % (array_article[idx][0], y, array_article[idx][2]))

    # scores = model.evaluate(x_test, y_test, batch_size=100)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))




if __name__ == "__main__":
    args = argumentparser.ArgumentParser()
    main(args)