import random
import os

# list [FOLLOWING] -> [[PRECEDING1, PROB1], PRECEDING2, PROB2]  
ABS_PATH = "C:/Users/Kuba/Documents/magisterka/ResponseBiasDL"
QUESTIONS = 30
ORDERS = 500
RESPONSE_BIAS_MIN = -0.06
RESPONSE_BIAS_MAX = 0.05
RESPONSE_BIAS_CHANCE = 0.3
RESPONDERS_PER_ORDER = 1000

def create_dirs():
    DATA_DIR = ABS_PATH + "/data"
    QUESTIONS_ORDERS_DIR = DATA_DIR + "/questions_orders"
    PROCESSED_DATA = DATA_DIR + "/processed_data.csv"

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(QUESTIONS_ORDERS_DIR):
        os.makedirs(QUESTIONS_ORDERS_DIR)
    if os.path.exists(PROCESSED_DATA):
        os.remove(PROCESSED_DATA)
        with open(PROCESSED_DATA, 'w') as file:
            for i in range(1, QUESTIONS + 1):
                file.write("o" + str(i) + ",")
            for i in range(1, QUESTIONS + 1):
                file.write("o" + str(i) + ",")
            file.write("output\n")

def generate_questions_response_biases():
    with open(ABS_PATH + "/data/questions_response_biases.txt", "w") as file:
        file.write("following preceding probability_influence\n")
        for q1 in range(1, QUESTIONS + 1):
            for q2 in range(1, QUESTIONS + 1):
                if q1 != q2:
                    roll = random.random()
                    if roll < RESPONSE_BIAS_CHANCE:
                        probability_influence = random.uniform(RESPONSE_BIAS_MIN, RESPONSE_BIAS_MAX)
                        file.write(f"{q1} {q2} {probability_influence:.2f}\n")

def generate_questions_orders():
    for i in range(ORDERS):
        order = [i+1 for i in range(QUESTIONS)]
        with open(ABS_PATH + f"/data/questions_orders/questions_order_{i}.txt", "w") as file:
            random.shuffle(order)
            for elem in order:
                file.write(str(elem) + "\n")

def get_questions_response_biases():
    question_dependencies = [[0.0] * (QUESTIONS + 1) for _ in range(QUESTIONS + 1)]
    #print(question_dependencies)
    with open(ABS_PATH + "/data/questions_response_biases.txt", "r") as file:
        for line in file.readlines()[1:]:
            values = line[:-1].split(" ")
            question_dependencies[int(values[0])][int(values[1])] = float(values[2])

    return question_dependencies

def get_questions_orders():
    orders = []
    for order in sorted(os.listdir(ABS_PATH + "/data/questions_orders/")):
        with open(ABS_PATH + "/data/questions_orders/" + order, "r") as file:
            list = []
            for elem in file.readlines():
                list.append(int(elem))

            orders.append(list)

    return orders

def get_questions_chances():
    chances = [0.0]
    with open(ABS_PATH + "/data/questions_chances.txt", "r") as file:
        for elem in file:
            chances.append(float(elem))

    return chances

def calc_response_bias(current_question, past_questions, questions_response_biases):
    response_bias = 0.0

    for past_question in past_questions:
        response_bias += questions_response_biases[current_question][past_question]
    return response_bias

def generate_answers_for_order(order, question_chances_with_bias):
    with open(ABS_PATH + f"/data/processed_data.csv", "a") as file:
        for i in range(0, len(order)):
            suborder = [0] * (len(order) - i - 1)
            suborder = suborder + order[0:i+1]

            for elem in suborder:
                file.write(str(elem) + ",")
            file.write(str(question_chances_with_bias[suborder[-1]]))
            file.write("\n")

def generate_answers():
    questions_response_biases = get_questions_response_biases()
    questions_orders = get_questions_orders()
    questions_chances = get_questions_chances()

    for order in questions_orders:
        quenstion_chances_with_bias = questions_chances.copy()
        past_questions = []
        for question in order:
            print(question, " ")
            chance = questions_chances[question]
            chance += calc_response_bias(question, past_questions, questions_response_biases)

            #print(question, f"{chance:.2f}")
            past_questions.append(question)
            quenstion_chances_with_bias[question] = chance

        generate_answers_for_order(order, quenstion_chances_with_bias)

    print(questions_response_biases)
    print(questions_orders)
    print(questions_chances)

if __name__ == "__main__":
    create_dirs()
    generate_questions_response_biases()
    generate_questions_orders()
    generate_answers()