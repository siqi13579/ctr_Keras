import os
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
print('nfm_dot_embedding')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


file_path = "nfm_dot_embedding.h5"
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
        # fm embeddings
        field = []
        embed = Embedding(nums, 10, input_length=1, trainable=True)(input)
        reshape = Reshape((10,))(embed)
        field_embedding.append(reshape)
        # fm embeddings
    #######fm layer##########
    inner_product = []
    for i in range(field_cnt):
        for j in range(i + 1, field_cnt):
            tmp = dot([field_embedding[i], field_embedding[j]], axes=1)
            inner_product.append(tmp)
    #######dnn layer##########
    embed_layer = concatenate(field_embedding + inner_product, axis=-1)
    embed_layer = Dense(64)(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dense(64)(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dense(1)(embed_layer)
    ########linear layer##########
    preds = Activation('sigmoid')(embed_layer)
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
checkpoint = ModelCheckpoint(
    file_path, save_weights_only=True, verbose=1, save_best_only=True)
early = EarlyStopping(monitor="val_loss", patience=2)
callbacks_list = [early, checkpoint]  # early
model = base_model(cols, train, test)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test), callbacks=callbacks_list, shuffle=False)
