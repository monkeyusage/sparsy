import JSON, StatFiles, CSV
using DataFrames, ProgressBars

function get_replacements(classes::Vector{T})::Dict{T, UInt64} where {T<:Number}
    # map unique elements to ints
    replacements = Dict{T, UInt64}()
    for (tclass, nclass) in enumerate(unique(classes))
        replacements[nclass] = tclass
    end
    replacements
end

function extract_matrix(data::DataFrame, )

function main()
    config = JSON.parsefile("data/config.json")
    input_file = config["input_data"]
    output_file = config["output_data"]
    iter_size = config["year_iteration"]

    data = DataFrame(StatFiles.load(input_file))
    data = data[:, ["year", "firm", "nclass"]]
    data[!, "year"] = map(UInt16, data[!, "year"])
    data[!, "nclass"] = map(UInt32, data[!, "nclass"])

    # replace nclass by tclass and save mapping to json
    replacements = get_replacements(data[!, "nclass"])
    open("data/replacements.json", "w") do f
        write(f, JSON.json(replacements, 4))
    end

    replace!(data[!, "nclass"], replacements...) # replace nclass to be tclass
    rename!(data, "nclass" => "tclass") # rename nclass to tclass
    sort!(data, "year") # sort by year

    CSV.write("data/intermediate.csv", data)

    for (index, _) in ProgressBar(enumerate(years))
        year_set = index+iter_size < length(years) ? Set(years[index:index+iter_size]) : Set(years[index:end])
        subset = data[data[!, "year"] => in(year_set) , :]
        freq = freqtable(df, :firm, :tclass)
        firms = names(freq)[1]
    end
end

export main