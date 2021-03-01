import pandas as pd
import numpy as np
import h5py
import os
import argparse
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

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
        "--path",
        type=str,
        default="./",
        help="The path of the project."
    )
    parser.add_argument(
        "--cell_line",
        type=str,
        default="GM12878",
        help="The cell line of dataset."
    )
    return parser.parse_args()

def data_process(args):
    # negative data
    data_neg = pd.read_csv(os.path.join(args.path, 'data/data_set', args.cell_line, 'neg_data_sequence.txt'), sep='\t')
    # data_neg=shuffle(data_neg)

    labels_neg = np.array(data_neg['labels'])
    seqs_neg = np.array(data_neg['seq'])
    fri_neg = np.array(data_neg['fri(x)'])
    gnm_neg = np.array(data_neg['GNM'])
    std_fri_neg = np.array(data_neg['std(fri)'])
    std_gnm_neg = np.array(data_neg['std(gnm)'])
    chr_neg = np.array(data_neg['chr'])

    X_neg = list()
    for i in range(seqs_neg.shape[0]):
        seq = seqs_neg[i]
        X_neg.append(np.array([base_mapping[c] for c in seq]))
    X_neg = np.array(X_neg)
    y_neg = labels_neg.reshape((-1, 1))

    print(X_neg.shape)
    print(y_neg.shape)

    # positive data
    data_pos = pd.read_csv(os.path.join(args.path, 'data/data_set', args.cell_line, 'pos_data_sequence.txt'), sep='\t')
    # data_pos=shuffle(data_pos)

    labels_pos = np.array(data_pos['labels'])
    seqs_pos = np.array(data_pos['seq'])
    fri_pos = np.array(data_pos['fri(x)'])
    gnm_pos = np.array(data_pos['GNM'])
    std_fri_pos = np.array(data_pos['std(fri)'])
    std_gnm_pos = np.array(data_pos['std(gnm)'])
    chr_pos = np.array(data_pos['chr'])

    X_pos = list()
    for i in range(labels_pos.shape[0]):
        seq = seqs_pos[i]
        X_pos.append(np.array([base_mapping[c] for c in seq]))
    X_pos = np.array(X_pos)
    y_pos = labels_pos.reshape((-1, 1))

    print(X_pos.shape)
    print(y_pos.shape)

    return (y_neg, X_neg, fri_neg, gnm_neg, std_fri_neg, std_gnm_neg, chr_neg,
           y_pos, X_pos, fri_pos, gnm_pos, std_fri_pos, std_gnm_pos, chr_pos)

def save_final_data(args):
    (y_neg, X_neg, fri_neg, gnm_neg, std_fri_neg, std_gnm_neg, chr_neg,
     y_pos, X_pos, fri_pos, gnm_pos, std_fri_pos, std_gnm_pos, chr_pos)=data_process(args)

    neg_len=X_neg.shape[0]
    pos_len=X_pos.shape[0]

    X_train=np.concatenate((X_neg[0:int(neg_len*0.8)],X_pos[0:int(pos_len*0.8)]),axis=0)
    y_train=np.concatenate((y_neg[0:int(neg_len*0.8)],y_pos[0:int(pos_len*0.8)]),axis=0)
    fri_train=np.concatenate((fri_neg[0:int(neg_len*0.8)],fri_pos[0:int(pos_len*0.8)]),axis=0)
    gnm_train=np.concatenate((gnm_neg[0:int(neg_len*0.8)],gnm_pos[0:int(pos_len*0.8)]),axis=0)
    std_fri_train=np.concatenate((std_fri_neg[0:int(neg_len*0.8)],std_fri_pos[0:int(pos_len*0.8)]),axis=0)
    std_gnm_train=np.concatenate((std_gnm_neg[0:int(neg_len*0.8)],std_gnm_pos[0:int(pos_len*0.8)]),axis=0)
    chr_train=np.concatenate((chr_neg[0:int(neg_len*0.8)],chr_pos[0:int(pos_len*0.8)]),axis=0)

    X_val=np.concatenate((X_neg[int(neg_len*0.8):int(neg_len*0.9)],X_pos[int(pos_len*0.8):int(pos_len*0.9)]),axis=0)
    y_val=np.concatenate((y_neg[int(neg_len*0.8):int(neg_len*0.9)],y_pos[int(pos_len*0.8):int(pos_len*0.9)]),axis=0)
    fri_val=np.concatenate((fri_neg[int(neg_len*0.8):int(neg_len*0.9)],fri_pos[int(pos_len*0.8):int(pos_len*0.9)]),axis=0)
    gnm_val=np.concatenate((gnm_neg[int(neg_len*0.8):int(neg_len*0.9)],gnm_pos[int(pos_len*0.8):int(pos_len*0.9)]),axis=0)
    std_fri_val=np.concatenate((std_fri_neg[int(neg_len*0.8):int(neg_len*0.9)],std_fri_pos[int(pos_len*0.8):int(pos_len*0.9)]),axis=0)
    std_gnm_val=np.concatenate((std_gnm_neg[int(neg_len*0.8):int(neg_len*0.9)],std_gnm_pos[int(pos_len*0.8):int(pos_len*0.9)]),axis=0)
    chr_val=np.concatenate((chr_neg[int(neg_len*0.8):int(neg_len*0.9)],chr_pos[int(pos_len*0.8):int(pos_len*0.9)]),axis=0)

    X_test=np.concatenate((X_neg[int(neg_len*0.9):],X_pos[int(pos_len*0.9):]),axis=0)
    y_test=np.concatenate((y_neg[int(neg_len*0.9):],y_pos[int(pos_len*0.9):]),axis=0)
    fri_test=np.concatenate((fri_neg[int(neg_len*0.9):],fri_pos[int(pos_len*0.9):]),axis=0)
    gnm_test=np.concatenate((gnm_neg[int(neg_len*0.9):],gnm_pos[int(pos_len*0.9):]),axis=0)
    std_fri_test=np.concatenate((std_fri_neg[int(neg_len*0.9):],std_fri_pos[int(pos_len*0.9):]),axis=0)
    std_gnm_test=np.concatenate((std_gnm_neg[int(neg_len*0.9):],std_gnm_pos[int(pos_len*0.9):]),axis=0)
    chr_test=np.concatenate((chr_neg[int(neg_len*0.9):],chr_pos[int(pos_len*0.9):]),axis=0)

    print("X_train shape:",X_train.shape)
    print("y_train shape:",y_train.shape)
    print("fri_train shape:",fri_train.shape)
    print("gnm_train shape:",gnm_train.shape)
    print("std_fri_train shape:",std_fri_train.shape)
    print("std_gnm_train shape:",std_gnm_train.shape)
    print("chr_train shape:",chr_train.shape)

    print("X_val shape:",X_val.shape)
    print("y_val shape:", y_val.shape)
    print("fri_val shape:", fri_val.shape)
    print("gnm_val shape:", gnm_val.shape)
    print("std_fri_val shape:", std_fri_val.shape)
    print("std_gnm_val shape:", std_gnm_val.shape)
    print("chr_val shape:", chr_val.shape)

    print("X_test shape:",X_test.shape)
    print("y_test shape:",y_test.shape)
    print("fri_test shape:",fri_test.shape)
    print("gnm_test shape:",gnm_test.shape)
    print("std_fri_test shape:",std_fri_test.shape)
    print("std_gnm_test shape:",std_gnm_test.shape)
    print("chr_test shape:",chr_test.shape)

    print("Data saving...")
    with h5py.File(os.path.join(args.path, 'data/final_set', args.cell_line, 'data_train.h5'), 'w') as data_train:
        data_train['X_train']=np.array(X_train).astype(np.int8)
        data_train['y_train']=np.array(y_train).astype(np.int8)
        data_train['fri_train']=np.array(fri_train).astype(np.float32)
        data_train['gnm_train']=np.array(gnm_train).astype(np.float32)
        data_train['std_fri_train']=np.array(std_fri_train).astype(np.float32)
        data_train['std_gnm_train']=np.array(std_gnm_train).astype(np.float32)

    with h5py.File(os.path.join(args.path, 'data/final_set', args.cell_line, 'data_val.h5'), 'w') as data_val:
        data_val['X_val']=np.array(X_val).astype(np.int8)
        data_val['y_val']=np.array(y_val).astype(np.int8)
        data_val['fri_val']=np.array(fri_val).astype(np.float32)
        data_val['gnm_val']=np.array(gnm_val).astype(np.float32)
        data_val['std_fri_val']=np.array(std_fri_val).astype(np.float32)
        data_val['std_gnm_val']=np.array(std_gnm_val).astype(np.float32)

    with h5py.File(os.path.join('./data/final_set',args.cell_line,'data_test.h5'), 'w') as data_test:
        data_test['X_test']=np.array(X_test).astype(np.int8)
        data_test['y_test']=np.array(y_test).astype(np.int8)
        data_test['fri_test']=np.array(fri_test).astype(np.float32)
        data_test['gnm_test']=np.array(gnm_test).astype(np.float32)
        data_test['std_fri_test']=np.array(std_fri_test).astype(np.float32)
        data_test['std_gnm_test']=np.array(std_gnm_test).astype(np.float32)
    
    final_set_path=os.path.join(args.path, 'data/final_set', args.cell_line)
    if not os.path.exists(final_set_path):
        os.makedirs(final_set_path)
    
    chr_train_inf=pd.DataFrame()
    chr_train_inf['chr_train']=chr_train
    chr_train_inf.to_csv(os.path.join(final_set_path, 'chr_train.txt'),index=False,header=True,sep='\t')

    chr_val_inf=pd.DataFrame()
    chr_val_inf['chr_val']=chr_val
    chr_val_inf.to_csv(os.path.join(final_set_path, 'chr_val.txt'),index=False,header=True,sep='\t')

    chr_test_inf=pd.DataFrame()
    chr_test_inf['chr_test']=chr_test
    chr_test_inf.to_csv(os.path.join(final_set_path, 'chr_test.txt'),index=False,header=True,sep='\t')

    print("Finish")

if __name__=='__main__':
    args = parse_args()
    save_final_data(args)
