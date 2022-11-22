'''
Keysight TrueQ - Noisy output extrapolation (NOX)
'''

import trueq as tq
import numpy as np
import trueq.compilation as tqc
import qiskit as qs

# The following option applies noisy output extrapolation (NOX); an error mitigation protocol from the True-Q library.

def nox(circuit, backend, n_compilations=20):

    tq.interface.qiskit.set_from_backend(backend)
    circuit_tq, metadata = tq.interface.qiskit.to_trueq_circ(circuit)
    
    circuit_tq = tqc.UnmarkCycles().apply(circuit_tq)
    
    nox_collection_trueq = tq.make_nox(circuit_tq)
    nox_collection_qiskit = [tq.interface.qiskit.from_trueq_circ(c, metadata=metadata) for c in nox_collection_trueq]
     
    return nox_collection_qiskit, nox_collection_trueq