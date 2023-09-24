import numpy as np
from qiskit import Aer
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, TwoLocal, EfficientSU2
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQR
from qiskit.algorithms.optimizers import SPSA
from qiskit.tools.events import TextProgressBar
import time
from utils import load_data, load_test_data

start = 0
stop = 0
iter_num = 1

def progress_callback(eval_count, obj_value):
    global iter_num, start, stop

    stop = time.time()
    print(f"Iter: {iter_num}, Time: {stop - start}, obj_val: {obj_value}")
    start = time.time()
    iter_num += 1

# 1. Quantum Encoding

feature_dimension = 20 # time_steps
feature_map = ZZFeatureMap(feature_dimension=feature_dimension)

#feature_map.decompose().draw(output="mpl", filename="feature.png")
# 2. Variational Quantum Circuit
#var_form = TwoLocal(num_qubits=feature_dimension, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz', reps=1)
var_form = EfficientSU2(num_qubits=feature_dimension, reps=1)
var_form.decompose().draw(output="mpl", filename="var_form.png")

# Sample dataset (replace with your time series data)
BATCH_SIZE = 32


data = np.random.rand(BATCH_SIZE, feature_dimension)
labels = np.random.rand(BATCH_SIZE)

train_data_inputs, train_data_outputs = load_data()
test_data_inputs, test_data_outputs = load_test_data()

# 3. Training the Model

optimizer = SPSA(maxiter=100)

backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=42, seed_transpiler=42)

vqr = VQR(feature_map=feature_map, ansatz=var_form, optimizer=optimizer, quantum_instance=quantum_instance, callback=progress_callback)

a.e()

start = time.time()
vqr.fit(train_data_inputs, train_data_outputs)

predicted_labels = vqr.predict(testing_data)

print(predicted_labels)