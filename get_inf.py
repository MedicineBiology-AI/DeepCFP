import numpy as np
import pandas as pd
import h5py
import os
import argparse

import predict as prediction
from keras.utils import to_categorical

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell_line",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The name of testing model."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00001,
        help="The name of testing model."
    )
    return parser.parse_args()

def get_inf(args):
    print("Loading data...")

    with h5py.File(os.path.join('./data', 'final_set', args.cell_line, 'data_test.h5'), 'r') as data_test:
        X_test = np.array(data_test['X_test'])
        y_test = np.array(data_test['y_test'])
        fri = np.array(data_test['fri_test'])
        gnm = np.array(data_test['gnm_test'])
        std_fri = np.array(data_test['std_fri_test'])
        std_gnm = np.array(data_test['std_gnm_test'])
    
    chr_inf = pd.read_table(os.path.join('./data', 'final_set', args.cell_line, 'chr_test.txt'))
    chr_test = chr_inf['chr_test']
        
    print("X_test shape:",X_test.shape)
    print("y_test shape:",y_test.shape)

    pred = prediction.predict(X_test,y_test,args)
    pred = np.array(pred)
    pred = pred.reshape(-1)
    print('prediction shape:',pred.shape)
    
    labels = y_test.reshape(-1)
    
    data = pd.DataFrame()
    data['chr'] = chr_test
    data['labels'] = labels
    data['FRI'] = fri
    data['GNM'] = gnm
    data['std(fri)'] = std_fri
    data['std(gnm)'] = std_gnm
    data['prediction'] = pred
    
    save_path=os.path.join('./compare', args.cell_line, args.cell_line+'_'+args.model_name+'_datacmp.txt')
    
    data.to_csv(save_path,index=False,header=True,sep='\t')
    
if __name__ == "__main__":
    args = parse_args()
    get_inf(args)