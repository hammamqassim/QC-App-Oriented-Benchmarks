#################################################################
## Post-processor module: True-Q error mitigation handlers ##
#################################################################

import trueq as tq
import qiskit as qk
import copy

def nox_handler(result, active_circuit):
    
    result_copy = copy.copy(result)
    nox_collection_trueq = active_circuit["transformer_metadata"].copy()
    counts = result.get_counts()
    shots = result.results[0].shots

    for idx, circuit in enumerate(nox_collection_trueq):
        circuit.results = flip_bits(counts[idx])
        
    mitigated_counts = generate_mitigated_counts(nox_collection_trueq, shots)
    mitigated_counts = flip_bits(mitigated_counts)
    
    res_copy = copy.copy(result.results[0])
    ret = copy.copy(result)
    
    res_copy.header.name = active_circuit["qc"].name 
    
    res_copy.data.counts = mitigated_counts
    ret.results = [res_copy]
    
    return ret


##################################################################
## Helper functions ##
##################################################################

def flip_bits(counts_dict):
    new_dict = dict()
    for key, val in counts_dict.items():
        new_dict[key[::-1]] = val
    return new_dict

def generate_mitigated_counts(tq_collection, shots):
    n_qubits = tq_collection.n_sys
    res_dict = dict()
    for k in range(2 ** n_qubits):
        bs = f'{k:0{n_qubits}b}'
        prob = tq_collection.fit(observables=[bs]).estimates[0]._values[0]
        res_dict[bs] = prob * shots if prob > 0 else 0
    return res_dict