import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Dropout, Concatenate, Flatten, BatchNormalization
import pandas as pd

from utils import load_data, load_test_data

QUESTIONS = 30

BATCH_SIZE = 32
DATA_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data/train_data.csv"
TEST_DATA_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data/test_data.csv"

def load_test_data():
    df = pd.read_csv(TEST_DATA_PATH)

    o_columns = []
    for i in range(QUESTIONS):
        o_columns.append("o" + str(i + 1))

    output_column = ['output']

    o_data = df[o_columns].values
    o_data = o_data.reshape((len(o_data), QUESTIONS, 1))  # reshaping to [samples, time steps, features]

    output_data = df[output_column].values

    dataset = tf.data.Dataset.from_tensors(({"lstm_input": o_data}))
    return dataset

def load_data(path):
    df = pd.read_csv(path)

    o_columns = []
    for i in range(QUESTIONS):
        o_columns.append("o" + str(i + 1))

    output_column = ['output']

    o_data = df[o_columns].values
    output_data = df[output_column].values

    o_data = o_data.reshape((len(o_data), QUESTIONS, 1))  # reshaping to [samples, time steps, features]

    dataset = tf.data.Dataset.from_tensors(({"lstm_input": o_data}, output_data))
    return dataset

def get_model_lstm():
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(QUESTIONS, 1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))
    return model

def get_model():
    input_order = Input(shape=(QUESTIONS)) # batch, time steps, features
    x = Dense(512, activation='swish')(input_order)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='swish')(x)
    x = Flatten()(x)
    output = Dense(1)(x)

    model = Model(inputs=[input_order], outputs=output)
    return model

def main():
    model = get_model_lstm()
    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    model.summary()

    dataset = load_data(DATA_PATH)
    print(dataset)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='ckpt/best_model.h5',
        monitor='loss',  # Monitor the validation loss
        save_best_only=True,  # Save only the best model
        save_weights_only=True,  # Save only the model weights, not the entire model
        verbose=1
    )

    model.fit(dataset, epochs=500, batch_size=64, shuffle=True, workers=24, use_multiprocessing=True, callbacks=[checkpoint_callback])

    test_dataset = load_data(TEST_DATA_PATH)
    print(model.evaluate(dataset))
    print(model.evaluate(test_dataset))

if __name__ == "__main__":
    main()