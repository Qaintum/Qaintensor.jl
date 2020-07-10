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

include("decompose.jl")
export
    decompose!

include("tensor_circuit.jl")
export
    tensor_circuit!

include("helper.jl")
export
    tn_to_ssa,
    cgc_to_ssa,
    tn_graph_creation

include("breadth_search_contraction.jl")
export
    optimal_contraction_order

include("greedy_contraction.jl")
export
    greedy_contraction_order

include("contract.jl")
export
    contract

include("contraction_order.jl")
export
    optimize_contraction_order!

include("catn_contraction.jl")
export
    catn_contraction_cost

    include("swap_contraction.jl")
    export
        swap_contraction_cost

include("cost_estimation.jl")
    export
        cost_estimate

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
