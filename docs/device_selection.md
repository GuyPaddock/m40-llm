# CUDA device selection and M40 guardrails

This project targets Tesla M40 (sm_52). By default, the runtime will select device 0 or respect CUDA_VISIBLE_DEVICES. Tests and benches offer helpers for explicitly selecting an M40-compatible context.

- Runtime helper: ctx_m40() creates a CudaContext selecting a sm_52-capable device if available
- Guard: require_sm52(ctx) will skip CUDA tests if the active device is not sm_52
- Env override: set M40LLM_FORCE_M40=1 to enforce sm_52 at runtime; non‑matching devices will cause an error

Notes
- CUDA_VISIBLE_DEVICES affects device enumeration seen by the program
- Non‑CUDA builds keep tests green via cfg gates; CUDA builds require nvcc
- When using benches/tests, prefer cuda_env::ctx_m40() for uniform selection
