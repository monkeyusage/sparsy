function tclass_corr(matrix::Array{Float32, 2})::Array{Float32, 2}
    """
    # correlation over N * M matrix that return M * M correlation matrix
    this one does not use threading since M is usually a small numner
    """
    var = matrix'matrix
    base_var = copy(var)
    s = size(var)[1] # arrays are 1 indexed
    @inbounds @simd for i in 1:s
        @inbounds @simd for j in 1:s 
            var[i, j] = var[i,i] == 0 || var[j,j] == 0 ? 1 : var[i, j] / (sqrt(base_var[i,i]) * sqrt(base_var[j,j]))
        end
    end
    return var
end

function logging(use_logger::Bool, year::Integer, metric::String)::Union{NTuple{2, Nothing}, Tuple{Task, Channel}}
    # first create buffered channel to throw values into, buffered because otherwise we might run out of memory while writing
    # create simple consumer, schedule it and return channel & task for later clean up
    if !use_logger
        return nothing, nothing
    end

    chan = Channel{Pair{NTuple{3, Int}, Float32}}(100_000)
    open("data/tmp/intermediate_$(metric)_$year.csv", "w") do io
        write(io, "year,firm1,firm2,class,value\n");
    end

    function consumer()
        # this function takes on values from pipe until it's empty
        # writes them to tmp file
        open("data/tmp/intermediate_$(metric)_$year.csv", "a") do io
            while true
                try
                    pair = take!(chan)
                    (firm1, firm2, class), value =  pair
                    write(io, "$year,$firm1,$firm2,$class,$value\n");
                catch
                    break
                end
            end
        end
    end

    # wrap it in a task to run in the background
    task = @task consumer()
    schedule(task) # launch it
    
    return task, chan
end


function dot_zero(
    matrix::Array{Float32, 2},
    weights::Array{Float32, 1},
    use_logger::Bool=false,
    year::Integer=1970,
    metric::String=""
)::Array{Float32}
    """
    # vectorized version of the following operations with M (n, m) => NM (n, n) => n
    out = matrix * matrix' => creates a matrix we cannot store in RAM
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    """
    N, M = size(matrix)
    out = Array{Float32, 1}(undef, N)

    bg_task, channel = logging(use_logger, year, metric)
    
    Threads.@threads for i in 1:N
        total = zero(Float32)
        remainder = zero(Float32)
        @inbounds for ii in 1:N
            if (i == ii) continue end
            @inbounds for j in 1:M
                value = matrix[i, j] * matrix[ii, j]
                total += value * weights[ii]
            end
            if use_logger # put pair into the channel, this will block if the buffer is full
                put!(channel, (i, ii) => total - remainder) # remove the remainder to extract value attributed to ii
                remainder = total
            end
        end
        @inbounds out[i] = total
    end

    if use_logger
        # clean up resources
        close(channel)
        wait(bg_task)
    end

    return out
end

function mahalanobis(
    biggie::Array{Float32, 2},
    small::Array{Float32, 2},
    weights::Array{Float32, 1},
    use_logger::Bool=false,
    year::Integer=1970,
    metric::String=""
)::Array{Float32}
    """
    # vectorized version of the following operations
    out = biggie * (small * biggie')
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    """
    N, M = size(biggie)
    out = Array{Float32}(undef, N)
    
    bg_task, channel = logging(use_logger, year, metric)

    Threads.@threads for i in 1:N
        total = Float32(0)
        remainder = Float32(0)
        @inbounds for ii in 1:N
            if ii == i continue end
            @inbounds for j in 1:M
                # logging firm pair and the value associated to it
                value = biggie[i, j] * small[j, ii]
                total += value * weights[ii]
            end
            if use_logger # put pair into the channel, this will block if the buffer is full
                put!(channel, (i, ii) => total - remainder) # remove the remainder to extract value attributed to ii
                remainder = total
            end
        end
        @inbounds out[i] = total
    end

    if use_logger
        # clean up resources
        close(channel)
        wait(bg_task)
    end

    return out
end
