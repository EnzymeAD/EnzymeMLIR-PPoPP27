"""Select MPAS reference or freshly collected data without depending on the working directory."""
function mpas_plot_paths(args = String[]; env = ENV, root = dirname(@__DIR__))
    source = get(env, "PLOT_SOURCE", "reference")
    input = get(env, "PLOT_INPUT", "")
    output = get(env, "PLOT_OUTPUT_DIR", "")
    i = 1
    while i <= length(args)
        flag = args[i]
        flag in ("--source", "--input", "--output-dir") ||
            error("Unknown option $flag. Use --source reference|new, --input CSV, --output-dir DIR")
        i < length(args) || error("Missing value for $flag")
        value = args[i + 1]
        if flag == "--source"
            source = value
        elseif flag == "--input"
            input = value
        else
            output = value
        end
        i += 2
    end
    source in ("reference", "new") || error("Invalid source '$source': expected reference or new")
    # Preserve the notebook's original RESULTS_DIR override when no new input was supplied.
    if isempty(input) && haskey(env, "RESULTS_DIR")
        input = joinpath(env["RESULTS_DIR"], "summary.csv")
    end
    if isempty(input)
        input = source == "reference" ?
            joinpath(root, "results", "mpas4dvar_binomial_sweep.csv") :
            joinpath(root, "results", "binomial_sweep", "summary.csv")
    end
    isempty(output) && (output = joinpath(root, "results", "plots", "mpas", source))
    input, output = abspath(input), abspath(output)
    isfile(input) || error("No MPAS $source data at $input. Run the sweep first or pass --input CSV.")
    (; source, input, output)
end
