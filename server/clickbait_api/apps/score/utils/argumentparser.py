from __future__ import print_function

import argparse
import os
from django.conf import settings

APP_DIR = "apps/score/"
DATASET_PATH = "data/source/"

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.join(settings.BASE_DIR, APP_DIR, "data/"),
                        help='')
    parser.add_argument('--mission', type=str, default='',
                        help='')
    parser.add_argument('--dataset', type=str, default='processing_article.csv',
                        help='')
    parser.add_argument('--source_dir', type=str, default='source/',
                        help='')
    parser.add_argument('--dest_dir', type=str, default='dest/',
                        help='')
    parser.add_argument('--morpheme_dir', type=str, default='morpheme/run.sh',
                        help='')
    parser.add_argument('--morpheme_dest_dir', type=str, default='morpheme_dest/',
                        help='')
    parser.add_argument('--id_dir', type=str, default='',
                        help='')
    parser.add_argument('--headline_dir', type=str, default='',
                        help='')
    parser.add_argument('--body_dir', type=str, default='',
                        help='')
    parser.add_argument('--paragraph_id_dir', type=str, default='',
                        help='')
    parser.add_argument('--paragraph_dir', type=str, default='',
                        help='')
    parser.add_argument('--sentence_dir', type=str, default='',
                        help='')
    parser.add_argument('--train_id_dir', type=str, default='train_data/mission_1_id.csv',
                        help='')
    parser.add_argument('--train_headline_dir', type=str, default='train_data/mission_1_head.csv',
                        help='')
    parser.add_argument('--train_sentence_dir', type=str, default='train_data/mission_1_sentence.csv',
                        help='')

    parser.add_argument('--embedding_file_path', type=str, default='embedding_model/true_article_notag_original_embeddings.npz',
                        help='path to file for embedding vectors')
    parser.add_argument('--vocab_file_path', type=str, default='embedding_model/true_article_notag_original_map.json',
                        help='path to file for embedding vocabulary')


    parser.add_argument('--mission1_model_dir', type=str, default='model/mission_1',
                        help='directory to store models')
    parser.add_argument('--mission2_model_dir', type=str, default='model/mission_2_model_1_2.h5',
                        help='directory to store models')
    parser.add_argument('--output_dir', type=str, default='output/',
                        help='directory to store models')


    parser.add_argument('--nb_words', type=int, default=20000,
                        help='Number of words to keep from the dataset')
    parser.add_argument('--max_sequence_len', type=int, default=30,
                        help='Maximum input sequence length')
    parser.add_argument('--max_headline_len', type=int, default=1,
                        help='Maximum input headline length')
    parser.add_argument('--max_body_len', type=int, default=29,
                        help='Maximum input body length')
    parser.add_argument('--len_labels_index', type=int, default=1,
                        help='length of labels')

    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data to be used for train')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Fraction of data to be used for validation')


    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of the embedding space to be used')
    parser.add_argument('--blstm_hidden_dim', type=int, default=100,
                        help='Dimension of the embedding space to be used')


    parser.add_argument('--model_name', type=str, default='cnn-non-static',
                        help='Name of the model variant, from the CNN Sentence '
                             'Classifier paper. Possible values are cnn-rand, cnn-static'
                             'cnn-non-static. If nothing is specified, it uses the arguments'
                             'passed to the script to define the hyperparameters. To add'
                             'your own model, pass model_name as self, define your model in'
                             'app/model/model.py and invoke from model_selector function.')


    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--num_patience', type=int, default=5,
                        help='number of patience')

    args, unknown = parser.parse_known_args()

    return args
