using Documenter
using Splinter

makedocs(;
         repo=Remotes.GitHub("palday", "Splinter.jl"),
         sitename="Splinter",
         doctest=true,
         checkdocs=:exports,
         warnonly=[:cross_references],
         format= Documenter.HTML(; edit_link="main"),
         pages=["index.md", 
                "api.md"])


deploydocs(; repo="github.com/palday/Splinter.jl.git", 
           devbranch="main",
           push_preview=true)
