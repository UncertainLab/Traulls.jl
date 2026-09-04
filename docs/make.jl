using Documenter
using Traulls

makedocs(
    sitename = "Traulls.jl",
    format = Documenter.HTML(),
    modules = [Traulls],
    pages = [
        "Home" => "index.md",
        "Method" => "method.md",
        "Tutorials" => "tutorials.md",
        "Reference" => "reference.md"
    ],
    warnonly = [:missing_docs]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
