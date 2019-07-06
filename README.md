# EMbench

* Julia package behind blog post on [EM benchmarking](https://floswald.github.io/post/em-benchmarks/)

## Installation

1. [Download julia](https://julialang.org/downloads/)
2. start julia. you see this:
    ```
    ➜  julia
                   _
       _       _ _(_)_     |  Documentation: https://docs.julialang.org
      (_)     | (_) (_)    |
       _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
      | | | | | | |/ _` |  |
      | | |_| | | | (_| |  |  Version 1.1.0 (2019-01-21)
     _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
    |__/                   |
    

    julia> 
    ```
3. Hit the `]` key to switch to package manager mode. the prompt switches to 
    ```
    (v1.1) pkg>
    ```
4. Download this package by pasting this into the `(v1.1) pkg>` prompt and hitting enter.
    ```julia
    dev https://github.com/floswald/EMbench.jl.git
    ```
5. After this is done, hit backspace or `ctrl-c` to go back to standard `julia>` prompt.
    ```julia
    julia> cd(joinpath(DEPOT_PATH[1],"dev","EMbench"))  # go to the location of EMbench
    ```
6. Go back to package mode again: type `]`. then:
    ```julia
    (v1.1) pkg> activate .     # tell pkg manager to modify current directory
    (EMbench) pkg> instantiate    # download all dependencies
    ```
7. Done! :tada: Now try it out. Go back to command mode with `ctrl-c`
    ```julia
    julia> using EMbench
    julia> EMbench.allbm()   # runs full benchmark
    ```