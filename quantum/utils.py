import pandas as pd

QUESTIONS = 20

BATCH_SIZE = 32
DATA_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data_20_questions/train_data.csv"
TEST_DATA_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data_20_questions/test_data.csv"

def load_test_data():
    df = pd.read_csv(TEST_DATA_PATH)

    o_columns = []
    for i in range(QUESTIONS):
        o_columns.append("o" + str(i + 1))

    output_column = ['output']

    inputs = df[o_columns].values
    inputs = inputs.reshape((len(inputs), QUESTIONS))  # reshaping to [samples, time steps, features]

    outputs = df[output_column].values

    return inputs, outputs

def load_data():
    df = pd.read_csv(TEST_DATA_PATH)

    o_columns = []
    for i in range(QUESTIONS):
        o_columns.append("o" + str(i + 1))

    output_column = ['output']

    inputs = df[o_columns].values
    inputs = inputs.reshape((len(inputs), QUESTIONS))  # reshaping to [samples, time steps, features]

    outputs = df[output_column].values
    
    return inputs, outputs