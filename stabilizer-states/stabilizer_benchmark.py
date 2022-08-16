"""
Stabilizer State Benchmark Program - Qiskit
"""

import sys
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, quantum_info

sys.path[1:1] = [ "_common", "_common/qiskit" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit" ]
import execute as ex
import metrics as metrics

np.random.seed(0)

verbose = False

# saved circuits for display
QC_ = None
# Uf_ = None

############### Circuit Definitions

def StabilizerCircuit(num_qubits):
    cliff = quantum_info.random_clifford(num_qubits)
    cliff_circuit = cliff.to_circuit()
    pauli = random_stabilizer(cliff)
    circuit = append_measurement(cliff_circuit, pauli)
    return circuit

def StabilizerCircuit2(num_qubits, depth=10):
    
    circuit = QuantumCircuit(num_qubits)

    for layer in range(depth):
        for qubit in range(num_qubits):
            gate = quantum_info.random_clifford(1).to_circuit()
            circuit = circuit.compose(gate, [qubit])
        #circuit.barrier()
        for qubit in range(layer%2, num_qubits - 1, 2):
            gate = quantum_info.random_clifford(2).to_circuit()
            circuit = circuit.compose(gate, [qubit, qubit + 1])
        #circuit.barrier()
    
    cliff = quantum_info.Clifford(circuit)
    pauli = random_stabilizer(cliff)
    circuit = append_measurement(circuit, pauli)
    
    return circuit

def random_stabilizer(cliff):
    n_sys = cliff.num_qubits
    m = np.random.randint(1, 2 ** n_sys)
    random_bitstring = f'{{0:0{n_sys}b}}'.format(m)
    stabilizer = (quantum_info.Pauli("".join(n_sys * ['I'])), 1)
    for idx, bit in enumerate(random_bitstring):
        if int(bit) == 1:
            p = quantum_info.Pauli(cliff.stabilizer.to_labels()[idx])
            pauli, phase = quantum_info.Pauli.sgn_prod(stabilizer[0], p)
            stabilizer = (pauli, phase * stabilizer[1])
    return (stabilizer[0].to_label(), stabilizer[1])

def append_measurement(circuit, pauli):
    stabilizer = pauli[0]
    phase = pauli[1]
    
    circuit_copy = circuit.copy()
    circuit_copy.barrier()
    
    mmt_qubits = []

    for idx, pauli in enumerate(reversed(stabilizer)):
        if pauli == 'X':
            circuit_copy.h(idx)
            mmt_qubits += [idx]
        if pauli == 'Y':
            circuit_copy.sdg(idx)
            circuit_copy.h(idx)
            mmt_qubits += [idx]
        if pauli == 'Z':
            mmt_qubits += [idx]

    if phase == -1:
        circuit_copy.x(mmt_qubits[0])

    cl = ClassicalRegister(size=len(mmt_qubits))
    circuit_copy.add_register(cl)

    circuit_copy.barrier()

    for idx, qubit in enumerate(mmt_qubits):
        circuit_copy.measure(qubit, cl[idx])
    
    return circuit_copy

############### Result Data Analysis

# Analyze and print measured results
# Expected result is always an even bitstring, so fidelity calc is simple

def analyze_and_print_result(qc, result):
    
    # obtain counts from the result object
    counts = result.get_counts(qc)
    #n_shots = result.results[0].shots
    
    # create the ideal and experimental distributions over parity
    
    ### measuring a stabilizer of a stabilizer state returns a 
    ### bitstring with even parity with probaility %100
    ideal_dist = dict({'0': 1,  '1': 0}) 
    exp_dist = dict({'0': 0,  '1': 0})
    
    for key, value in counts.items():
        parity = sum(list(map(int, key))) % 2
        exp_dist[f'{parity}'] += value

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(exp_dist, ideal_dist)
    # fidelity = exp_dist['0']
        
    return counts, fidelity

################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=3, max_qubits=6, max_circuits=3, num_shots=100,
        backend_id='qasm_simulator', method = 1, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):

    print("Stabilizer State Benchmark Program - Qiskit")

    # validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = max(2, min_qubits)
    
    transform_qubit_group = False
    
    # Initialize metrics module
    metrics.init_metrics()
    
    # Define custom result handler
    def execution_handler (qc, result, num_qubits, type , num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result)
        metrics.store_metric(num_qubits, type, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # for noiseless simulation, set noise model to be None
    # ex.set_noise_model(None)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1):
    
        num_circuits = max_circuits
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # loop over different random Cliffords
        for j in range(num_circuits):
            
            # create the circuit for given qubit size and store time metric
            ts = time.time()
            qc = StabilizerCircuit(num_qubits) if method==1 else StabilizerCircuit2(num_qubits)
            metrics.store_metric(num_qubits, j, 'create_time', time.time()-ts)

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc, num_qubits, j, shots=num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    
    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - Stabilizer States - Qiskit")

# if main, execute method
if __name__ == '__main__': run()
   
