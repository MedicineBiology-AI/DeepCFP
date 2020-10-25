import numpy as np
import h5py
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell_name",
        type=str,
        default="GM12878",
        help="The cell type of dataset."
    )
    return parser.parse_args()

def load_final_data(args):
    with h5py.File(os.path.join('./data/final_set',args.cell_name,'data_train.h5'),'r') as data_train:
        X_train=np.array(data_train['X_train'])
        y_train=np.array(data_train['y_train'])

    with h5py.File(os.path.join('./data/final_set',args.cell_name,'data_val.h5'),'r') as data_val:
        X_val=np.array(data_val['X_val'])
        y_val=np.array(data_val['y_val'])
        
    with h5py.File(os.path.join('./data/final_set',args.cell_name,'data_test.h5'),'r') as data_test:
        X_test=np.array(data_test['X_test'])
        y_test=np.array(data_test['y_test'])
    
    return X_train,y_train,X_val,y_val,X_test,y_test
    
def load_cross_validation_data(args):
    with h5py.File(os.path.join('./data/cross_validation',args.cell_name,'data_train.h5'),'r') as data_train:
        X_train=np.array(data_train['X_train'])
        y_train=np.array(data_train['y_train'])

    with h5py.File(os.path.join('./data/cross_validation',args.cell_name,'data_val.h5'),'r') as data_val:
        X_val=np.array(data_val['X_val'])
        y_val=np.array(data_val['y_val'])
    
    return X_train,y_train,X_val,y_val

if __name__ == "__main__":
    args = parse_args()
    X_train,y_train,X_val,y_val,X_test,y_test=load_final_data(args)
    #X_train,y_train,X_val,y_val=load_cross_validation_data(args)
    
    print("X_train shape:",X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:",X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:",y_test.shape)
    
