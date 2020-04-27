
struct Tensor

    data::Array

    # TODO: add some meta-information like quantum numbers for tensor legs
end


ndims(t::Tensor) = Base.ndims(t.data)
