import os

QUESTIONS = 5
ORDERS = 10
RESPONDERS_PER_ORDER = 1000

BASE_DIR = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL/data/questions_answers"

for order_num in range(ORDERS):
    answer_stats = [0] * QUESTIONS
    order_dir = BASE_DIR + f"/order_{order_num}"

    for responder in range (RESPONDERS_PER_ORDER):
        with open(order_dir + f"/responder_{responder}.txt") as file:
            i = 0
            for answer in file.readlines():
                answer_stats[i] += int(answer)
                i += 1

    print("order", order_num)
    print(answer_stats)