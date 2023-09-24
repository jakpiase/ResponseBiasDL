import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Dropout, Concatenate, Flatten, BatchNormalization
import pandas as pd

QUESTIONS = 30

BATCH_SIZE = 32
TRAIN_DATA_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data/train_data.csv"
TEST_DATA_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data/test_data.csv"
WEIGHTS_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/dl/ckpt/best_model_earlier.h5"

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

    model.load_weights(WEIGHTS_PATH)

    train_dataset = load_data(TRAIN_DATA_PATH)
    test_dataset = load_data(TEST_DATA_PATH)

    train_results = model.evaluate(train_dataset)
    test_results = model.evaluate(test_dataset)

    print("Train MSE:", train_results[0])
    print("Train MAE:", train_results[1])
    print("Test MSE:", test_results[0])
    print("Test MAE:", test_results[1])

if __name__ == "__main__":
    main()