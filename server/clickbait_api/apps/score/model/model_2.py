from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Merge, Embedding, Input, LSTM, Bidirectional, Lambda, Reshape, Conv1D, concatenate
from keras.optimizers import Adadelta, RMSprop

# model - mission_1_model_2
# head - 50, body - 500



def model_selector(args, embedding_matrix):
    '''Method to select the model to be used for classification'''
    if (args.model_name.lower() != 'self'):
        return _predefined_model(args, embedding_matrix)




def _predefined_model(args, embedding_matrix):
    '''function to use one of the predefined models (based on the paper)'''
    (filtersize_list, min_conv_size_list, number_of_filters_per_filtersize, pool_length_list,
     dropout_list, optimizer, use_embeddings, embeddings_trainable) \
        = _param_selector(args)



    # Headline
    print('Defining title model.')
    if (use_embeddings):
        embedding_layer = Embedding(args.nb_words,
                                    args.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=args.max_headline_len,
                                    trainable=embeddings_trainable)
    else:
        embedding_layer = Embedding(args.nb_words,
                                    args.embedding_dim,
                                    weights=None,
                                    input_length=args.max_headline_len,
                                    trainable=embeddings_trainable)



    input_node_title = Input(shape=(args.max_headline_len, args.embedding_dim))
    # embedd = embedding_layer(input_node_title)
    dropout_1 = Dropout(dropout_list[0], input_shape=(args.max_headline_len, args.embedding_dim))(input_node_title)
    conv_list_title = []
    for index, filtersize in enumerate(filtersize_list):
        nb_filter = number_of_filters_per_filtersize[index]
        # pool_length = pool_length_list[index]print(x)
        conv = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')(dropout_1)
        # pool = MaxPooling1D(pool_length=pool_length)(conv)
        # print(pool)
        # flatten = Reshape((100, args.embedding_dim))(conv)
        # print(flatten)
        # flatten = Flatten()(pool)
        crop = _crop(2, 0, min_conv_size_list[0])(conv)
        # print(dir(conv))
        # print(conv.eval)
        conv_list_title.append(crop)

    concate = concatenate(conv_list_title)
    new_row = (int(concate.shape[1]) * int(concate.shape[2])) / int(args.blstm_hidden_dim)
    reshape = Reshape((int(new_row), args.blstm_hidden_dim))(concate)
    blstm = Bidirectional(LSTM(args.blstm_hidden_dim))(reshape)
    title_out = Dropout(dropout_list[1])(blstm)


    # Body
    print('Defining body model.')
    if (use_embeddings):
        embedding_layer = Embedding(args.nb_words,
                                    args.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=args.max_body_len,
                                    trainable=embeddings_trainable)
    else:
        embedding_layer = Embedding(args.nb_words,
                                    args.embedding_dim,
                                    weights=None,
                                    input_length=args.max_body_len,
                                    trainable=embeddings_trainable)



    input_node_body = Input(shape=(args.max_body_len, args.embedding_dim))
    # embedd = embedding_layer(input_node_body)
    dropout_1 = Dropout(dropout_list[0], input_shape=(args.max_body_len, args.embedding_dim))(input_node_body)
    conv_list_title = []
    for index, filtersize in enumerate(filtersize_list):
        nb_filter = number_of_filters_per_filtersize[index]
        # pool_length = pool_length_list[index]
        conv = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')(dropout_1)
        # print()
        # pool = MaxPooling1D(pool_length=pool_length)(conv)
        # print(pool)
        # flatten = Reshape((100, args.embedding_dim))(conv)
        # print(flatten)
        # flatten = Flatten()(pool)
        crop = _crop(2, 0, min_conv_size_list[1])(conv)
        # print(dir(conv))
        # print(conv.eval)
        conv_list_title.append(crop)

    concate_2 = concatenate(conv_list_title)
    new_row = (int(concate_2.shape[1]) * int(concate_2.shape[2])) / int(args.blstm_hidden_dim)
    reshape_2 = Reshape((int(new_row), args.blstm_hidden_dim))(concate_2)
    blstm_2 = Bidirectional(LSTM(args.blstm_hidden_dim))(reshape_2)
    body_out = Dropout(dropout_list[1])(blstm_2)



    # Final model
    print('Defining final model.')
    concate_3 = concatenate([title_out, body_out])
    x = Dense(100)(concate_3)
    x = Dropout(dropout_list[2])(x)
    main_out = Dense(args.len_labels_index, activation='sigmoid')(x)
    model = Model(inputs=[input_node_title, input_node_body], outputs=main_out)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model



def _predefined_model_2(args, embedding_matrix):
    '''function to use one of the predefined models (based on the paper)'''
    (filtersize_list, min_conv_size_list, number_of_filters_per_filtersize, pool_length_list,
     dropout_list, optimizer, use_embeddings, embeddings_trainable) \
        = _param_selector(args)



    # Headline
    print('Defining title model.')
    if (use_embeddings):
        embedding_layer = Embedding(args.nb_words,
                                    args.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=args.max_headline_len,
                                    trainable=embeddings_trainable)
    else:
        embedding_layer = Embedding(args.nb_words,
                                    args.embedding_dim,
                                    weights=None,
                                    input_length=args.max_headline_len,
                                    trainable=embeddings_trainable)


    input_node_title = Input(shape=(args.max_headline_len, args.embedding_dim))
    conv_list_title = []
    for index, filtersize in enumerate(filtersize_list):
        nb_filter = number_of_filters_per_filtersize[index]
        # pool_length = pool_length_list[index]
        conv = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')(input_node_title)

        # print()

        # pool = MaxPooling1D(pool_length=pool_length)(conv)
        # print(pool)
        # flatten = Reshape((100, args.embedding_dim))(conv)
        # print(flatten)
        # flatten = Flatten()(pool)
        x = _crop(2, 0, min_conv_size_list[0])(conv)
        # print(dir(conv))
        # print(conv.eval)
        conv_list_title.append(x)


    if (len(filtersize_list) > 1):
        out_title = Merge(mode='concat')(conv_list_title)
    else:
        out_title = conv_list_title[0]

    new_row = (int(out_title.shape[1]) * int(out_title.shape[2])) / int(args.blstm_hidden_dim)
    reshape_title = Reshape((int(new_row), args.blstm_hidden_dim))(out_title)
    graph_title = Model(input=input_node_title, output=reshape_title)
    # graph = Model(input=input_node, output=out)

    model_title = Sequential()
    model_title.add(embedding_layer)
    model_title.add(Dropout(dropout_list[0], input_shape=(args.max_headline_len, args.embedding_dim)))
    model_title.add(graph_title)
    model_title.add(Bidirectional(LSTM(args.blstm_hidden_dim)))
    model_title.add(Dropout(dropout_list[1]))






    # Body
    print('Defining body model.')
    if (use_embeddings):
        embedding_layer = Embedding(args.nb_words,
                                    args.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=args.max_body_len,
                                    trainable=embeddings_trainable)
    else:
        embedding_layer = Embedding(args.nb_words,
                                    args.embedding_dim,
                                    weights=None,
                                    input_length=args.max_body_len,
                                    trainable=embeddings_trainable)





    input_node_body = Input(shape=(args.max_body_len, args.embedding_dim))
    conv_list_body = []
    for index, filtersize in enumerate(filtersize_list):
        nb_filter = number_of_filters_per_filtersize[index]
        # pool_length = pool_length_list[index]
        conv = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')(input_node_body)
        # pool = MaxPooling1D(pool_length=pool_length)(conv)
        # print(pool)
        # flatten = Reshape((100, args.embedding_dim))(conv)
        # print(flatten)
        # flatten = Flatten()(pool)
        x = _crop(2, 0, min_conv_size_list[1])(conv)
        conv_list_body.append(x)


    if (len(filtersize_list) > 1):
        out_body = Merge(mode='concat')(conv_list_body)
    else:
        out_body = conv_list_body[0]

    new_row = (int(out_body.shape[1]) * int(out_body.shape[2])) / int(args.blstm_hidden_dim)
    reshape_body = Reshape((int(new_row), args.blstm_hidden_dim))(out_body)
    graph_body = Model(input=input_node_body, output=reshape_body)
    # graph = Model(input=input_node, output=out)

    model_body = Sequential()
    model_body.add(embedding_layer)
    model_body.add(Dropout(dropout_list[0], input_shape=(args.max_body_len, args.embedding_dim)))
    model_body.add(graph_body)
    model_body.add(Bidirectional(LSTM(args.blstm_hidden_dim)))
    model_body.add(Dropout(dropout_list[1]))






    # Final model
    print('Defining final model.')
    model_final = Sequential()
    model_final.add(Merge([model_title, model_body], mode='concat'))
    model_final.add(Dense(100))
    model_final.add(Dropout(dropout_list[2]))
    model_final.add(Dense(args.len_labels_index, activation='sigmoid'))
    model_final.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model_final

def _param_selector(args):
    '''Method to select parameters for models defined in Convolutional Neural Networks for
        Sentence Classification paper by Yoon Kim'''
    filtersize_list = [2, 3, 7]
    min_conv_size_list = [args.max_headline_len - max(filtersize_list) + 1, args.max_body_len - max(filtersize_list) + 1, args.max_sequence_len - max(filtersize_list) + 1]
    number_of_filters_per_filtersize = [100, 100, 100]
    pool_length_list = [2, 2, 2]
    dropout_list = [0.6, 0.6, 0.6]
    optimizer = Adadelta(clipvalue=3)
    use_embeddings = True
    embeddings_trainable = False

    if (args.model_name.lower() == 'cnn-rand'):
        print("embedding method: cnn-rand")
        use_embeddings = False
        embeddings_trainable = True
    elif (args.model_name.lower() == 'cnn-static'):
        print("embedding method: cnn-static")
        pass
    elif (args.model_name.lower() == 'cnn-non-static'):
        print("embedding method: cnn-non-static")
        embeddings_trainable = True
    else:
        filtersize_list = [3, 4, 5]
        number_of_filters_per_filtersize = [150, 150, 150]
        pool_length_list = [2, 2, 2]
        dropout_list = [0.25, 0.5]
        optimizer = RMSprop(lr=args.learning_rate, decay=args.decay_rate,
                            clipvalue=args.grad_clip)
        use_embeddings = True
        embeddings_trainable = True
    return (filtersize_list, min_conv_size_list, number_of_filters_per_filtersize, pool_length_list,
            dropout_list, optimizer, use_embeddings, embeddings_trainable)



def _crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, start: end, :]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return Lambda(func)

