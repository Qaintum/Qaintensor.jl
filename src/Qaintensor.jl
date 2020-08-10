module Qaintensor

using Qaintessent
using TensorOperations


include("tensor.jl")
export
    Tensor

include("tensor_network.jl")
export
    GeneralTensorNetwork,
    TensorNetwork,
    Summation,
    contract

include("decompose.jl")
export
    decompose!

include("tensor_circuit.jl")
export
    tensor_circuit!

include("svd.jl")
export
    contract_svd

include("mps.jl")
export
    MPS,
    OpenMPS,
    PeriodicMPS,
    ClosedMPS,
    contract_svd_mps,
    check_mps

include("mpo.jl")
export
    MPO,
    circuit_MPO,
    apply_MPO

include("switch.jl")
export
    switch!

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
