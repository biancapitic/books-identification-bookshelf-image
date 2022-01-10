import os
import matplotlib.pyplot as plt
from keras import Input
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Lambda, Bidirectional, CuDNNLSTM, Dense
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from textDataGenerator import TextDataGenerator


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_model(input_shape, characters, max_label_len):
    width = input_shape[0] # 128
    height = input_shape[1] # 32

    inputs = Input(name="input_layer",shape=(height, width, 1))

    conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
    conv_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_3)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_4)
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(64, (2, 2), activation='relu')(pool_6)

    squeezed = Lambda(lambda x: tf.keras.backend.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(squeezed)
    blstm_2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(blstm_1)

    outputs = Dense(len(characters) + 1, name="outputs", activation='softmax')(blstm_2)

    labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func,
                      output_shape=(1,),
                      name='ctc')([outputs, labels, input_length, label_length])

    return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

def get_all_characters(train_db_path, test_db_path):

    words = []
    longest_word_size = 0
    for path in os.listdir(train_db_path):
        word = path.split("_")[1]
        words.append(word)
        longest_word_size = max(longest_word_size, len(str(word)))

    for path in os.listdir(test_db_path):
        word = path.split("_")[1]
        words.append(word)
        longest_word_size = max(longest_word_size, len(str(word)))

    vocab = set("".join(map(str, words)))

    return sorted(vocab), longest_word_size

def load_trained_model(input_shape,train_db_path, test_db_path, model_file_path) :

    characters, max_label_len = get_all_characters(train_db_path, test_db_path)

    model = build_model(input_shape, characters, max_label_len)

    pred_model = tf.keras.models.Model(model.get_layer(name="input_layer").input,
                                       model.get_layer(name="outputs").output)
    pred_model.load_weights(model_file_path)

    pred_model.summary()

    return pred_model, characters


def train_text_model(input_shape, batch_size, epochs, train_db_path, test_db_path, save_model_file_path):
    characters, max_label_len = get_all_characters(train_db_path, test_db_path)

    train_generator = TextDataGenerator(train_db_path, input_shape, batch_size, characters, max_label_len)
    validation_generator = TextDataGenerator(test_db_path, input_shape, batch_size, characters, max_label_len)

    model = build_model(input_shape, characters, max_label_len)

    pred_model = tf.keras.models.Model(model.get_layer(name="input_layer").input,
                                       model.get_layer(name="outputs").output)
    pred_model.summary()

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    checkpoint = ModelCheckpoint(filepath=save_model_file_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    callbacks_list = [checkpoint]

    history = model.fit(train_generator,
                        epochs=epochs,
                        batch_size=batch_size,
                        steps_per_epoch=len(train_generator.data) // batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator.data) // batch_size,
                        verbose=1,
                        callbacks=callbacks_list,
                        shuffle=True)
    return history, pred_model, characters

def plot_model_results(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


