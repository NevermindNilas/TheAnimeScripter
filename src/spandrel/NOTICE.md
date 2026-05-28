# Vendored spandrel

Vendored (in-repo) fork of [TNTwise/spandrel](https://github.com/TNTwise/spandrel)
at commit `e747f2713c0ae82654d0338c5b58ce5ab152ad04` (branch `adding_extra_archs`),
itself a fork of [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel).

Previously a git submodule; converted to tracked files for full in-repo control
of architecture performance work.

The `spandrel_extra_arches` package was **removed** — those architectures ship
under non-permissive / research-only licenses. Only the permissively-licensed
`libs/spandrel/spandrel/` package is retained. Per-architecture upstream licenses
remain in each `architectures/<arch>/__arch/LICENSE`; spandrel's own license is in
`LICENSE`.
