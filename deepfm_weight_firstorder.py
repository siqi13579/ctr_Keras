import os
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization, subtract
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
print('deepfm_weight_firstorder')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


file_path = "deepfm_weight_firstorder.h5"
batch_size = 256
epochs = 1

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
numer_columns = [col for col in train.columns if 'numer' in col]
cat_columns = [col for col in train.columns if 'cat' in col]
y_train = train['label'].values
y_test = test['label'].values
# ----------------------------------model-------------


def base_model(cat_columns, train, test):
    cat_num = len(cat_columns)
    field_cnt = cat_num
    cat_field_input = []
    field_embedding = []
    lr_embedding = []
    for cat in cat_columns:
        input = Input(shape=(1,))
        cat_field_input.append(input)
        nums = pd.concat((train[cat], test[cat])).max() + 1
        embed = Embedding(nums, 1, input_length=1, trainable=True)(input)
        reshape = Reshape((1,))(embed)
        lr_embedding.append(reshape)
        # fm embeddings
        field = []
        embed = Embedding(nums, 10, input_length=1, trainable=True)(input)
        reshape = Reshape((10,))(embed)
        field_embedding.append(reshape)
        # fm embeddings
    #######fm layer##########
    fm_square = Lambda(lambda x: K.square(x))(add(field_embedding))
    square_fm = add([Lambda(lambda x:K.square(x))(embed)
                     for embed in field_embedding])
    inner_product = subtract([fm_square, square_fm])
    inner_product = Lambda(lambda x: x * 0.5)(inner_product)
    #######dnn layer##########
    embed_layer = concatenate(field_embedding, axis=-1)
    embed_layer = Dense(64)(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dense(64)(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    ########linear layer##########
    lr_layer = concatenate(lr_embedding + [embed_layer, inner_product])
    lr_layer = Dense(1)(lr_layer)
    preds = Activation('sigmoid')(lr_layer)
    opt = Adam(0.001)
    model = Model(inputs=cat_field_input, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    return model
# training########################################3
cols = cat_columns + numer_columns
x_train = train[cols].values
x_test = test[cols].values
x_train = list(x_train.T)
x_test = list(x_test.T)
early = EarlyStopping(monitor="val_loss", patience=3)
checkpoint = ModelCheckpoint(
    file_path, save_weights_only=True, verbose=1, save_best_only=True)
callbacks_list = [early, checkpoint]  # early
model = base_model(cols, train, test)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test), callbacks=callbacks_list, shuffle=False)
