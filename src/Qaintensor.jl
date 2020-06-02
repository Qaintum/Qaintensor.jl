module Qaintensor

using Qaintessent
using TensorOperations


include("tensor.jl")
export
    Tensor

include("tensor_network.jl")
export
    TensorNetwork,
    Summation

include("contract.jl")
export
    contract_order,
    contract_rep,
    contract

include("decompose.jl")
export
    decompose!

include("tensor_circuit.jl")
export
    tensor_circuit!


# re-export definitions from Qaintessent.jl

# gates
export
    AbstractGate,
    X,
    Y,
    Z,
    XGate,
    YGate,
    ZGate,
    HadamardGate,
    SGate,
    TGate,
    SdagGate,
    TdagGate,
    RxGate,
    RyGate,
    RzGate,
    RotationGate,
    PhaseShiftGate,
    SwapGate,
    ControlledGate,
    controlled_not

# circuit
export
    AbstractCircuitGate,
    CircuitGate,
    single_qubit_circuit_gate,
    two_qubit_circuit_gate,
    controlled_circuit_gate,
    rdm,
    CircuitGateChain,
    MeasurementOps,
    Circuit

# apply
export
    apply

# models
export
    qft_circuit


end
