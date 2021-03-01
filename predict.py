import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import numpy as np
import h5py
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES']='0'

from keras.optimizers import Adam
import load_data as ld
import build_model as bm
from keras.utils import to_categorical

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

def predict(X,y,args):
    
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
    #model.summary()
    
    print("Model compiling...")
    opt=Adam(lr=args.learning_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    print('Loading Weights...')
    model.load_weights(os.path.join(args.path, 'weights', args.cell_line, args.cell_line+'_'+args.model_name+'.h5df'))
    
    print('Evaluating...')
    score=model.evaluate(X,y,batch_size=args.batch_size)
    print("Test loss:",score[0])
    print("Test accuracy:",score[1])
    
    print('Predicting...')
    pred=model.predict(X,batch_size=args.batch_size)
    print("prediction shape:",pred.shape)
    
    return pred

if __name__ == "__main__":
    args = parse_args()
    print("Loading data...")
    _,_,_,_,X_test,y_test=ld.load_final_data(args)
        
    print("X_train shape:",X_train.shape)
    print("X_test shape:",X_test.shape)
    print("y_train shape:",y_train.shape)
    print("y_test shape:",y_test.shape)
    print(predict(X_test,y_test,args))
    
    
