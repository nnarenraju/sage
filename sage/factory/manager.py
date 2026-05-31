"""
sage.factory.manager — removed.

CompileManager and the compiled training/validation blocks were removed when
SageVanillaTraining replaced the compiled pipeline.  The neural network model
is compiled externally with torch.compile(model, ...) before being passed to
SageVanillaTraining, which is where the meaningful speedup comes from.

This file is kept as an empty stub so that any stale imports fail gracefully
with a clear message rather than a ModuleNotFoundError.
"""
