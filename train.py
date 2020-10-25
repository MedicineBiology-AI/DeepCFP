import tensorflow as tf
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
'''
import numpy as np
import h5py
import os
import keras
import argparse

# os.environ['CUDA_VISIBLE_DEVICES']='0'

from keras.callbacks import Callback, ModelCheckpoint, TensorBoard

import load_data as ld
import build_model as bm

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
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="The epochs of training."
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

def main(args):
    epochs = args.epochs
    batch_size = args.batch_size

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = ld.load_final_data(args)
    # X_train,y_train,X_val,y_val=ld.load_cross_validation_data(args)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("Build model...")
    if (args.model_name=='model1'):
        model = bm.build_model1()
    elif (args.model_name=='model2'):
        model = bm.build_model2()
    elif (args.model_name == 'model3'):
        model = bm.build_model3()
    elif (args.model_name == 'model4'):
        model = bm.build_model4()
    elif (args.model_name == 'deepcfp'):
        model = bm.build_DeepCFP()

    model.summary()

    print("Model compiling...")
    opt = keras.optimizers.Adam(lr=args.learning_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    class get_result(Callback):
        def on_epoch_end(self, epoch, logs={}):
            # print('train_loss:',logs.get('loss'))
            # print('train_acc:',logs.get('acc'))
            print('val_loss:', logs.get('val_loss'))
            print('val_acc', logs.get('val_acc'))
            print('')

    result = get_result()

    checkpoint_path = os.path.join('./weights', args.cell_name, args.cell_name + '_' + args.model_name + '.h5df')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_acc',
                                 mode=max)

    tb = TensorBoard(log_dir=os.path.join('./logs', args.cell_name, args.cell_name + '_' + args.model_name))

    callbacks = [result, checkpoint, tb]

    print("Training...")
    history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    shuffle=True,
                    verbose=2,
                    callbacks=callbacks)

if __name__=="__main__":
    args = parse_args()
    main(args)
