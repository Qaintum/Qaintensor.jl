
struct Tensor

    data::Array

    # TODO: add some meta-information like quantum numbers for tensor legs
end


ndims(t::Tensor) = Base.ndims(t.data)
Base.size(t::Tensor) = size(t.data)
Base.conj(t::Tensor) = Tensor(conj(t.data))
