
"""
    Tensor

stores vector of tensors
"""
struct Tensor
    data::Array
    # TODO: add some meta-information like quantum numbers for tensor legs
end

Base.ndims(t::Tensor) = Base.ndims(t.data)

Base.reshape(t::Tensor, dim::Tuple) = Tensor(reshape(t.data,dim))
Base.reshape(t::Tensor, dim::Int...) = Tensor(reshape(t.data, Tuple(dim)))
Base.size(t::Tensor) = Base.size(t.data)

function Base.isapprox(t1::Tensor, t2::Tensor)
    all(t1.data .≈ t2.data)
end
