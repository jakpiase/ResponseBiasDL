import os
DATA_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data/processed_data.csv"

TRAIN_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data/train_data.csv"
TEST_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data/test_data.csv"

TEST_PART = 0.2

with open(DATA_PATH, "r") as file:
    lines = file.readlines()

    FIRST_TEST_ELEM = round(len(lines) * TEST_PART)
    with open(TRAIN_PATH, "w") as train_file:
        for line in lines[FIRST_TEST_ELEM:]:  
            train_file.write(line)

    with open(TEST_PATH, "w") as test_file:
        for line in lines[1:FIRST_TEST_ELEM]:
            test_file.write(line)
