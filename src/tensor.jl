
struct Tensor

    data::Array

    # TODO: add some meta-information like quantum numbers for tensor legs
end


ndims(t::Tensor) = Base.ndims(t.data)
Base.reshape(t::Tensor,dim::Tuple) =Tensor(Base.reshape(t.data,dim))
Base.reshape(t::Tensor,dim::Int...)=reshape(t, dim)
Base.permutedims(t::Tensor, dim::Array)=Tensor(Base.permutedims(t.data, dim))

function adjoint_tensor(T::Tensor, inlegs, outlegs)

    #TODO: implement adjoint when the inlegs and outlegs are not ordered as the first and last legs

    T = T.data
    dim_in, dim_out = size(T)[1:length(inlegs)], size(T)[length(inlegs)+1:end]
    newdims = (prod(dim_in), prod(dim_out))
    T = reshape(T, newdims)
    Tdagger = adjoint(T)
    Tdagger = reshape(Tdagger, reverse(newdims))
    Tdagger = reshape(Tdagger,dim_out..., dim_in...)
    return Tensor(Tdagger)
end

function isunitary(T::Tensor, inlegs, outlegs)

    #TODO: implement this when the inlegs and outlegs are not ordered as the first and last legs

    #TODO: implement for "non-square tensors" (number of inlegs different from outlegs)

    length(inlegs) == length(outlegs) || error("Number of incoming legs is different from outgoing legs")
    Tdagger = adjoint_tensor(T, inlegs, outlegs)

    T, Tdagger = T.data, Tdagger.data
    dim_in, dim_out = size(T)[1:length(inlegs)], size(T)[length(inlegs)+1:end]
    newdims = (prod(dim_in), prod(dim_out))
    T, Tdagger = reshape(T, newdims), reshape(Tdagger, reverse(newdims))

    return T*Tdagger ≈ I

end


function LinearAlgebra.ishermitian(T::Tensor)

    #TODO: implement this when the inlegs and outlegs are not ordered as the first and last legs

    n=Qaintensor.ndims(T)
    (n % 2) == 0 || error("Number of incoming legs is different from outgoing legs")

    return T.data ≈ adjoint_tensor(T, [1:n÷2...], [(n÷2+1):n...]).data
end
