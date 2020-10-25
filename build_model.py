from keras.models import Sequential, Model
from keras.layers import Input, Flatten
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional
from keras.layers import TimeDistributed, Reshape
from keras.layers import Add, Concatenate, Dot
from keras import regularizers

from attention import AttentionLayer

#Convolution param
filter_num=128
filter_length=200
    
#BILSTM param
lstm_units=64

#Dense param
danse_units=256

def build_model1():
    input_layer = Input(shape=(5000, 4))

    conv_layer1=Conv1D(filters=64,
                      kernel_size=20,
                      padding='same',
                      activation='relu',
                      strides=1,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      name='cnn1')(input_layer)
    
    bn1=BatchNormalization(name='bn1')(conv_layer1)
    
    conv_layer2=Conv1D(filters=128,
                      kernel_size=20,
                      padding='same',
                      activation='relu',
                      strides=1,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      name='cnn2')(bn1)
                      
    bn2=BatchNormalization(name='bn2')(conv_layer2)

    max_pool_layer = MaxPooling1D(pool_size=int(filter_length / 2),
                                  strides=int(filter_length / 2),
                                  padding='same',
                                  name='pooling')(bn2)

    bn3 = BatchNormalization(name='bn3')(max_pool_layer)

    bilstm_layer = Bidirectional(LSTM(units=lstm_units,
                                      return_sequences=True,
                                      kernel_regularizer=regularizers.l2(1e-5),
                                      bias_regularizer=regularizers.l2(1e-5)), name='lstm')(bn3)

    bn4 = BatchNormalization(name='bn4')(bilstm_layer)

    flatten = Flatten()(bn4)

    dense_layer = Dense(units=danse_units,
                        kernel_regularizer=regularizers.l2(1e-5),
                        bias_regularizer=regularizers.l2(1e-5),
                        activation='relu',
                        name='dense')(flatten)

    bn5 = BatchNormalization(name='bn5')(dense_layer)

    output_layer = Dense(units=1,
                         kernel_regularizer=regularizers.l2(1e-5),
                         bias_regularizer=regularizers.l2(1e-5),
                         activation='sigmoid',
                         name='classify')(bn5)

    model = Model(input=input_layer, output=output_layer)

    return model

def build_model2():
    input_layer = Input(shape=(5000, 4))

    conv_layer1=Conv1D(filters=64,
                      kernel_size=20,
                      padding='same',
                      activation='relu',
                      strides=1,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      name='cnn1')(input_layer)
    
    bn1=BatchNormalization(name='bn1')(conv_layer1)
    
    conv_layer2=Conv1D(filters=128,
                      kernel_size=20,
                      padding='same',
                      activation='relu',
                      strides=1,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      name='cnn2')(bn1)
                      
    bn2=BatchNormalization(name='bn2')(conv_layer2)

    max_pool_layer = MaxPooling1D(pool_size=int(filter_length / 2),
                                  strides=int(filter_length / 2),
                                  padding='same')(bn2)

    bn3 = BatchNormalization(name='bn3')(max_pool_layer)

    bilstm_layer = Bidirectional(LSTM(units=lstm_units,
                                      return_sequences=True,
                                      kernel_regularizer=regularizers.l2(1e-5),
                                      bias_regularizer=regularizers.l2(1e-5)), name='lstm')(bn3)

    bn4 = BatchNormalization(name='bn4')(bilstm_layer)

    attention_layer = AttentionLayer(name='attention')(bn4)

    bn5 = BatchNormalization(name='bn5')(attention_layer)

    dense_layer = Dense(units=danse_units,
                        kernel_regularizer=regularizers.l2(1e-5),
                        bias_regularizer=regularizers.l2(1e-5),
                        activation='relu',
                        name='dense')(bn5)

    bn6 = BatchNormalization(name='bn6')(dense_layer)

    output_layer = Dense(units=1,
                         kernel_regularizer=regularizers.l2(1e-5),
                         bias_regularizer=regularizers.l2(1e-5),
                         activation='sigmoid',
                         name='classify')(bn6)

    model = Model(input=input_layer, output=output_layer)

    return model

def build_model3():
    input_layer=Input(shape=(5000,4))
    
    conv_layer1=Conv1D(filters=64,
                      kernel_size=20,
                      padding='same',
                      activation='relu',
                      strides=1,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      name='cnn1')(input_layer)
    
    bn1=BatchNormalization(name='bn1')(conv_layer1)
    
    conv_layer2=Conv1D(filters=128,
                      kernel_size=20,
                      padding='same',
                      activation='relu',
                      strides=1,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      name='cnn2')(bn1)
                      
    bn2=BatchNormalization(name='bn2')(conv_layer2)
    
    reshape=Reshape((int(5000*2/filter_length),int(filter_length/2),128))(bn2)
    attention_pooling=TimeDistributed(AttentionLayer(),name='attentionPooling')(reshape)
    
    bn3=BatchNormalization(name='bn3')(attention_pooling)
    
    bilstm_layer=Bidirectional(LSTM(units=lstm_units,
                               return_sequences = True,
                               kernel_regularizer=regularizers.l2(1e-5),
                               bias_regularizer=regularizers.l2(1e-5)),name='bilstm')(bn3)
    
    bn4=BatchNormalization(name='bn4')(bilstm_layer)
    
    flatten = Flatten()(bn4)
    
    dense_layer=Dense(units=danse_units,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      activation='relu',
                      name='dense')(flatten)
    
    bn5=BatchNormalization(name='bn5')(dense_layer)
    #dp2=Dropout(0.5)(bn6)
    
    output_layer=Dense(units=1,
                     kernel_regularizer=regularizers.l2(1e-5),
                     bias_regularizer=regularizers.l2(1e-5),
                     activation='sigmoid',
                     name='classify')(bn5)
    
    model = Model(input=input_layer, output=output_layer)

    return model

def build_DeepCFP():
    input_layer=Input(shape=(5000,4))
    
    conv_layer1=Conv1D(filters=64,
                      kernel_size=20,
                      padding='same',
                      activation='relu',
                      strides=1,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      name='cnn1')(input_layer)
    
    bn1=BatchNormalization(name='bn1')(conv_layer1)
    
    conv_layer2=Conv1D(filters=128,
                      kernel_size=20,
                      padding='same',
                      activation='relu',
                      strides=1,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      name='cnn2')(bn1)
                      
    bn2=BatchNormalization(name='bn2')(conv_layer2)
    
    reshape=Reshape((int(5000*2/filter_length),int(filter_length/2),128))(bn2)
    attention_pooling=TimeDistributed(AttentionLayer(),name='attentionPooling')(reshape)
    
    bn3=BatchNormalization(name='bn3')(attention_pooling)
    
    bilstm_layer=Bidirectional(LSTM(units=lstm_units,
                               return_sequences = True,
                               kernel_regularizer=regularizers.l2(1e-5),
                               bias_regularizer=regularizers.l2(1e-5)),name='bilstm')(bn3)
    
    bn4=BatchNormalization(name='bn4')(bilstm_layer)
    
    attention_layer=AttentionLayer(name='attention')(bn4)
    
    bn5=BatchNormalization(name='bn5')(attention_layer)
    #dp1=Dropout(0.5)(bn5)
    
    dense_layer=Dense(units=danse_units,
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      activation='relu',
                      name='dense')(bn5)
    
    bn6=BatchNormalization(name='bn6')(dense_layer)
    #dp2=Dropout(0.5)(bn6)
    
    output_layer=Dense(units=1,
                     kernel_regularizer=regularizers.l2(1e-5),
                     bias_regularizer=regularizers.l2(1e-5),
                     activation='sigmoid',
                     name='classify')(bn6)
    
    model = Model(input=input_layer, output=output_layer)

    return model

if __name__ == "__main__":
    model=build_DeepCFP()
    model.summary()
