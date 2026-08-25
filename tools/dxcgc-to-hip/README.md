# dxcgc-to-hip

This importer converts the supported textual DxCGC subset emitted by the
`amdxcgc` parser into tensor-DPS HIP dialect MLIR.

```powershell
python dxcgc_to_hip.py input.mlir -o output.mlir
```

Run `python -m unittest test_dxcgc_to_hip.py` from this directory. Operations
with attributes, multiple results, or unsupported/fused semantics are rejected
until their lowering is implemented.
