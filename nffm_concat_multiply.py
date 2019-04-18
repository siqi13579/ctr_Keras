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
print('nffm_concat_multiply')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

file_path = "nffm_concat_multiply.h5"
batch_size = 1000
epochs = 1

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
numer_columns = [col for col in train.columns if 'numer' in col]
cat_columns = [col for col in train.columns if 'cat' in col]
y_train = train['label'].values
y_test = test['label'].values
# ----------------------------------model-------------


def base_model(cat_columns, numer_columns, train, test):
    cat_num = len(cat_columns)
    numer_num = len(numer_columns)
    cat_field_input = []
    cat_embedding = []
    for cat in cat_columns:
        input = Input(shape=(1,))
        cat_field_input.append(input)
        nums = pd.concat((train[cat], test[cat])).max() + 1
        field = []
        embed = Embedding(nums, 10 * cat_num, input_length=1,
                          trainable=True)(input)
        reshape = Reshape((cat_num, 10))(embed)
        cat_embedding.append(reshape)
    numer_field_input = []
    numer_embedding = []
    for cat in numer_columns:
        input = Input(shape=(1,))
        numer_field_input.append(input)
        nums = pd.concat((train[cat], test[cat])).max() + 1
        field = []
        embed = Embedding(nums, 10 * numer_num,
                          input_length=1, trainable=True)(input)
        reshape = Reshape((numer_num, 10))(embed)
        numer_embedding.append(reshape)

        # ffm embeddings
    #######ffm layer##########
    tmp1 = []
    tmp2 = []
    for i in range(cat_num):
        for j in range(i + 1, cat_num):
            tmp1.append(Lambda(lambda x: x[:, j, :])(cat_embedding[i]))
            tmp2.append(Lambda(lambda x: x[:, i, :])(cat_embedding[j]))
    tmp1 = concatenate(tmp1)
    tmp2 = concatenate(tmp2)
    cat_embed_layer = multiply([tmp1, tmp2])
    tmp1 = []
    tmp2 = []
    for i in range(numer_num):
        for j in range(i + 1, numer_num):
            tmp1.append(Lambda(lambda x: x[:, j, :])(numer_embedding[i]))
            tmp2.append(Lambda(lambda x: x[:, i, :])(numer_embedding[j]))
    tmp1 = concatenate(tmp1)
    tmp2 = concatenate(tmp2)
    numer_embed_layer = multiply([tmp1, tmp2])
    #######dnn layer##########
    embed_layer = concatenate([cat_embed_layer, numer_embed_layer])
    embed_layer = Dense(64)(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dense(64)(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dense(1)(embed_layer)
    ########linear layer##########
    lr_layer = embed_layer
    preds = Activation('sigmoid')(lr_layer)
    opt = Adam(0.001)
    model = Model(inputs=cat_field_input + numer_field_input, outputs=preds)
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
model = base_model(cat_columns, numer_columns, train, test)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test), callbacks=callbacks_list, shuffle=False)
