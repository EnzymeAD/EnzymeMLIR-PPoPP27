using Test
include(joinpath(@__DIR__, "..", "plotting", "mpas_paths.jl"))

@testset "MPAS plot input selection" begin
    mktempdir() do root
        ref = joinpath(root, "results", "mpas4dvar_binomial_sweep.csv")
        mkpath(dirname(ref))
        write(ref, "reference")
        paths = mpas_plot_paths(; root, env=Dict{String,String}())
        @test paths.input == ref
        @test paths.output == joinpath(root, "results", "plots", "mpas", "reference")
        @test_throws ErrorException mpas_plot_paths(["--source", "new"]; root, env=Dict())
        new = joinpath(root, "results", "binomial_sweep", "summary.csv")
        mkpath(dirname(new))
        write(new, "new data")
        paths = mpas_plot_paths(["--source", "new"]; root, env=Dict())
        @test paths.input == new
        @test paths.output == joinpath(root, "results", "plots", "mpas", "new")
        paths = mpas_plot_paths(["--source", "reference", "--input", new,
                                "--output-dir", joinpath(root, "custom output")]; root,
                               env=Dict("PLOT_SOURCE" => "new", "PLOT_INPUT" => ref))
        @test paths.source == "reference"
        @test paths.input == new
        @test endswith(paths.output, "custom output")
        @test mpas_plot_paths(; root, env=Dict("RESULTS_DIR" => dirname(new))).input == new
        @test_throws ErrorException mpas_plot_paths(["--source", "typo"]; root, env=Dict())
        @test_throws ErrorException mpas_plot_paths(["--input"]; root, env=Dict())
        @test_throws ErrorException mpas_plot_paths(["--typo"]; root, env=Dict())
    end
end
