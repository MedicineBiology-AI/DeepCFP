import pandas as pd
import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from keras.models import Model

import predict as prediction
from keras.optimizers import Adam
import load_data as ld
import build_model as bm

sub_length = 100

base_mapping = {
    'A': np.array([1, 0, 0, 0]),
    'G': np.array([0, 1, 0, 0]),
    'C': np.array([0, 0, 1, 0]),
    'T': np.array([0, 0, 0, 1]),
    'N': np.array([0, 0, 0, 0]),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell_name",
        type=str,
        default="GM12878",
        help="The cell type of dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepcfp",
        help="The name of testing model."
    )
    return parser.parse_args()

def get_peak(dic_path, data):
    start = np.array(data['start'])
    end = np.array(data['end'])
    chrr = np.array(data['chr'])
    sub_peak = np.zeros((data.shape[0], int(5000 / sub_length)))

    with open(dic_path, 'r') as f1:
        dic = eval(f1.read())
        # print(len(dic))

    for i in range(0, data.shape[0]):
        j = 0
        while (j < int(5000 / sub_length)):
            try:
                peak_start = start[i] - 1 + j * sub_length
                peak_end = dic[(chrr[i], peak_start)]
                while (peak_start < peak_end):
                    sub_peak[i][j] = 1
                    j = j + 1
                    peak_start = peak_start + sub_length
            except:
                j = j + 1
                continue

    return sub_peak

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def get_data_before_attention(model, X):
    print(X.shape)
    layer_name = 'bn4'
    intermediate_layer_model = Model(input=model.input,
                                     output=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(X)
    return intermediate_output

def get_attention_weights(model, data, X):
    layer_name = 'attention'
    intermediate_layer_model = Model(input=model.input,
                                     output=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(X)

    w = intermediate_layer_model.get_layer(layer_name).get_weights()[0]
    b = intermediate_layer_model.get_layer(layer_name).get_weights()[1]

    d = K.variable(data)
    w = K.variable(w)
    b = K.variable(b)

    f = K.tanh(K.dot(d, w) + b)
    a = K.softmax(K.batch_flatten(f))
    weights = np.array(K.eval(a))

    return weights

def save_attention_peak_information(args):
    data_pos = pd.read_csv(os.path.join('./data', 'data_set', args.cell_name, 'pos_data_sequence_v3.txt'), sep='\t')
    print("pos_data_sequence shape:", data_pos.shape)
    data_pos_test = data_pos[int(len(data_pos) * 0.9):]
    print(data_pos_test.shape)

    print("Build model...")
    if (args.model_name == 'model1'):
        model = bm.build_model1()
    elif (args.model_name == 'model2'):
        model = bm.build_model2()
    elif (args.model_name == 'model3'):
        model = bm.build_model3()
    elif (args.model_name == 'model4'):
        model = bm.build_model4()
    elif (args.model_name == 'deepcfp'):
        model = bm.build_DeepCFP()
    # model.summary()

    print("Model compiling...")
    opt = Adam(lr=1e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print('Loading Weights...')
    model.load_weights(os.path.join('./weights', args.cell_name, args.cell_name + '_' + args.md + '.h5df'))

    chrr = ['chr' + str(i) for i in range(1, 23)]
    chrr.append('chrX')

    for i in chrr:
        data = data_pos_test[data_pos_test['chr'].isin([i])]
        sub_peak = get_peak(os.path.join('./attention_peak', 'dict_peak_' + str(sub_length) + '.txt'), data)

        seqs_pos = np.array(data['seq'])
        labels_pos = np.array(data['labels'])

        print("Onehot sequences...")
        X_pos = list()
        for j in range(labels_pos.shape[0]):
            seq = seqs_pos[j]
            X_pos.append(np.array([base_mapping[c] for c in seq]))
        X_pos = np.array(X_pos)
        y_pos = labels_pos.reshape((-1, 1))
        print("Finish")

        print("X_pos shape:", X_pos.shape)
        print("y_pos shape:", y_pos.shape)
        print("sub peak shape:", sub_peak.shape)

        before_attention = get_data_before_attention(model, X_pos)
        weights = get_attention_weights(model, before_attention, X_pos)

        with h5py.File(os.path.join('./attention_peak', 'chr_data_' + str(sub_length), i + '_data.h5'), 'w') as f:
            f['sub_peak'] = sub_peak
            f['weights'] = weights

def attention_peak():
    sub_length = 100

    chrr = ['chr' + str(i) for i in range(1, 23)]
    chrr.append('chrX')

    ch = []
    low = []
    high = []
    mid1 = []
    mid2 = []

    for i in chrr:
        try:
            with h5py.File(
                    os.path.join('./attention_peak', 'chr_data_' + str(sub_length),
                                 i + '_data.h5'), 'r') as f:
                weights = np.array(f['weights'])
                peak = np.array(f['sub_peak'])
        except:
            continue

        peak = peak.reshape(-1)
        weights = weights.reshape(-1)
        index = np.argsort(weights)

        row = index.shape[0]
        peak_sum = np.sum(peak)

        l = peak[index[0:int(row * 0.25)]]
        m1 = peak[index[int(row * 0.25):int(row * 0.5)]]
        m2 = peak[index[int(row * 0.5):int(row * 0.75)]]
        h = peak[index[int(row * 0.75):row]]

        low.append(np.sum(l) / peak_sum)
        mid1.append(np.sum(m1) / peak_sum)
        mid2.append(np.sum(m2) / peak_sum)
        high.append(np.sum(h) / peak_sum)
        ch.append(i)
        '''
        print("0%-25% :",np.sum(l)/(row*0.25))
        print("25%-50% :",np.sum(m1)/(row*0.25))
        print("50%-75% :",np.sum(m2)/(row*0.25))
        print("75%-100% :",np.sum(h)/(row*0.25))
        '''

    plt.figure(figsize=(16, 8))
    bar_width = 0.15

    plt.bar(np.arange(23), low, label='0%-25%', color='#FEBF00', alpha=0.9, width=bar_width)
    plt.bar(np.arange(23) + bar_width + 0.05, mid1, label='25%-50%', color='#A5A5A5', alpha=0.9, width=bar_width)
    plt.bar(np.arange(23) + 2 * bar_width + 0.1, mid2, label='50%-75%', color='#ED7D31', alpha=0.9, width=bar_width)
    plt.bar(np.arange(23) + 3 * bar_width + 0.15, high, label='75%-100%', color='#4473C5', alpha=0.9, width=bar_width)
    plt.xlabel('Chromosome', fontsize=15)
    plt.ylabel('Percentage of transcription factor binding site', fontsize=15)
    plt.xticks(np.arange(23) + bar_width + 0.1, chrr, fontsize=12, rotation=20)
    plt.ylim([0.0, 0.4])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    save_attention_peak_information(args)
    attention_peak()
