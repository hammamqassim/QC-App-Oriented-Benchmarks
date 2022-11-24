'''
Keysight TrueQ - Randomized Compilation
'''

import trueq as tq
import numpy as np
import trueq.compilation as tqc
import qiskit as qs
import copy as copy
from collections import Counter

################################################################################################   
# RC transformers
################################################################################################

# The following option applies randomized compilation through the True-Q software. This compilation technique allocates the total shot budget over randomized implementations of the circuit, which typically results in suppressing the effect of coherent noise processes.

def full_rc(active_circuit, backend, n_compilations=20):

    circuit = copy.copy(active_circuit['transpiled_circuit'])  
    tq.interface.qiskit.set_from_backend(backend)
    circuit_tq, metadata = tq.interface.qiskit.to_trueq_circ(circuit)
    
    circuit_tq = tqc.UnmarkCycles().apply(circuit_tq)
    
    rc_collection = []
    for _ in range(n_compilations):
        c = compiler.compile(circuit_tq)
        c = split(c)
        c = mark_measurements(c)
        c = compiler_full_RC.compile(c)
        rc_collection += [tq.interface.qiskit.from_trueq_circ(c, metadata=metadata)]
        
    active_circuit['machine_circuits'] = rc_collection
    active_circuit['transformer_metadata'] = None
    active_circuit['machine_shots'] = int(active_circuit['shots'] / n_compilations)
    active_circuit['result_processor'] = aggregate

    
# The following applies a local variant of randomized compiling, in which twirling is only applied to active qubits in a given cycle, i.e. idling qubits are not twirled. For sparse circuits, this variant reduces the gate count and depth as compared to full RC, but is less effective in suppressing the effect of coherent crosstalk involving the idling qubits.

def local_rc(circuit, backend, n_compilations=20):

    tq.interface.qiskit.set_from_backend(backend)
    circuit_tq, metadata = tq.interface.qiskit.to_trueq_circ(circuit)
    
    circuit_tq = tqc.UnmarkCycles().apply(circuit_tq)
    
    rc_collection = []
    for _ in range(n_compilations):
        c = compiler.compile(circuit_tq)
        c = split(c)
        c = mark_measurements(c)
        c = compiler_local_RC.compile(c)
        rc_collection += [tq.interface.qiskit.from_trueq_circ(c, metadata=metadata)]
    
    active_circuit['machine_circuits'] = rc_collection
    active_circuit['transformer_metadata'] = None
    active_circuit['machine_shots'] = int(active_circuit['shots'] / n_compilations)
    active_circuit['result_processor'] = aggregate
    
################################################################################################   
# Result processors
################################################################################################

# The following computes the counts by summing them up, allowing for the case where
# <result> contains results from multiple circuits

def aggregate(result_object, active_circuit):
    
    result = copy.copy(result_object)
    result_obj = result.to_dict()
    
    # get the actual shots and convert to int if it is a string
    actual_shots = 0
    for experiment in result_obj["results"]:
        actual_shots += experiment["shots"]
    if type(actual_shots) is str:
        actual_shots = int(actual_shots)
        
    # aggregate        
    if type(result.get_counts()) == list:
        total_counts = dict()
        for count in result.get_counts():
            total_counts = dict(Counter(total_counts) + Counter(count))

        # replace the results array with an array containing only the first results object
        # then populate other required fields
        
        results = copy.copy(result.results[0])
        results.header.name = active_circuit["qc"].name     # needed to identify the original circuit
        results.shots = actual_shots
        results.data.counts = total_counts
        result.results = [ results ]
        
    return result

################################################################################################   
# Helper functions
################################################################################################
def split(circuit):
    new_circuit = tq.Circuit()
    for cycle in circuit.cycles[:-1]:
        if len(cycle.gates_single) > 0 and len(cycle.gates_multi) > 0:
            new_circuit.append(cycle.gates_single)
            new_circuit.append(cycle.gates_multi)
        else:
            new_circuit.append(cycle)
    
    if len(circuit.cycles[-1].gates) > 0 and len(circuit.cycles[-1].meas) > 0:
            new_circuit.append(circuit.cycles[-1].gates)
            new_circuit.append(circuit.cycles[-1].meas)
    else:
            new_circuit.append(circuit.cycles[-1])
            
    new_circuit = tq.compilation.UnmarkCycles().apply(new_circuit)
    return new_circuit

def mark_measurements(circuit):
    max_marker = max(cycle.marker for cycle in circuit.cycles)
    new_circuit = tq.Circuit()
    if circuit[-1].n_meas > 0 and circuit[-1].marker == 0:
        for cycle in circuit.cycles[:-1]:
            new_circuit.append(cycle)
        new_circuit.append(tq.Cycle(circuit[-1].operations, marker=max_marker+1))
        return new_circuit
    else:
        return circuit.copy()

################################################################################################   
# Compilers
################################################################################################
class Native1QSparse(tqc.Parallel):
    def __init__(self, factories, mode="ZXZXZ", angles=None, **_):

        replacements = [tqc.NativeExact(factories), 
                        tqc.Native1QMode(factories)
                       ]
        super().__init__(tqc.TryInOrder(replacements))

x90 = tq.config.GateFactory.from_hamiltonian("x90", [["X", 90]])
x180 = tq.config.GateFactory.from_hamiltonian("x180", [["X", 180]])
z = tq.config.GateFactory.from_hamiltonian("z", [["Z", "theta"]])
cnot = tq.config.GateFactory.from_matrix("cx", tq.Gate.cnot.mat)

factories = [x90, x180, z, cnot]

compiler = tq.Compiler(
    [
        tqc.Native2Q(factories),
        tqc.UnmarkCycles(),
        tqc.Merge(),
        tqc.RemoveId(),
        tqc.Justify(),
        tqc.RemoveEmptyCycle()
    ]
)

compiler_local_RC = tq.Compiler(
    [
        tqc.MarkBlocks(),
        tqc.RCLocal(),
        Native1QSparse(factories)
    ]
)

compiler_full_RC = tq.Compiler(
    [
        tqc.MarkBlocks(),
        tqc.RCCycle(),
        Native1QSparse(factories)
    ]
)

        
