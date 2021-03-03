module Qaintensor

using Qaintessent
using Qaintmodels
using TensorOperations

# re-export definitions from Qaintessent.jl
# gates
using Qaintessent:
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
using Qaintessent: 
    AbstractCircuitGate,
    CircuitGate,
    circuit_gate,
    rdm,
    MeasurementOperator,
    Circuit

# apply
using Qaintessent:
    apply

# models
using Qaintmodels:
    qft_circuit



include("tensor.jl")
export
    Tensor

include("tensor_network.jl")
export
    GeneralTensorNetwork,
    TensorNetwork,
    Summation
    
include("helper.jl")

include("contract.jl")
export
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
    extend_MPO,
    apply_MPO

include("switch.jl")
export
    switch!

include("network2graph.jl")
export
    network_graph,
    line_graph,
    optimize_contraction_order!

end
