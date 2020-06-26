try 
    run(`lualatex -v`)
catch ex
    @warn("Unable to run `lualatex -v`. Lualatex is required for visualizing LaserTag.")
    if Sys.islinux()
        @info("Attempting to install lualatex with apt-get.")
        # cmds = [`sudo apt-get install texlive-latex-base`, `sudo apt-get install texlive-binaries`]
        cmds = [`sudo apt-get install texlive-latex-extra`]
        for c in cmds
            @info("Running $c")
            run(c)
        end
    end
end

try
    run(`which pdf2svg`)
catch
    @warn("Unable to run `which pdf2svg`. pdf2svg is required for visualizing LaserTag.")
    if Sys.islinux()
        @info("Attempting to install pdf2svg with apt-get")
        cmd = `sudo apt-get install pdf2svg`
        @info("Running $cmd")
        run(cmd)
    end
end
