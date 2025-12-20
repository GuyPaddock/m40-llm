# Contributing

Thank you for contributing to m40-llm!

## Commit Messages

We use Conventional Commits. Format your commit header as:

```
<type>(<scope>): <subject>
```

Rules:
- type ∈ feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
- scope: optional; use components like cuda, kvcache, gguf, server, infer, allocator, attn, mlp, rmsnorm, build, ci, docs, tests
- subject: imperative mood, ≤ 50 chars, no period. Keep the subject crisp and specific.
- separate subject from body with a blank line
- wrap body at ~72 columns; explain what changed and why (not the steps you ran)
- Avoid vague subjects like "fmt" or "fixes" — say what changed and why
- Prefer `refactor(...)` over `chore(...)` for structural/API changes that do not alter behavior

## Building

### Non-CUDA Build
```bash
cargo build --no-default-features
```

### CUDA Build (Requires CUDA 12.x Toolkit)
```bash
cargo build --features cuda
```

### Testing
```bash
# Run all non-CUDA tests
cargo test --no-default-features --locked

# Run all CUDA tests (requires NVCC)
cargo test --features cuda --locked

# Run specific test with logging
RUST_LOG=debug cargo test test_name -- --nocapture
```

### Feature Flags
- `cuda`: Enables NVIDIA CUDA support (requires CUDA toolkit)
- `server`: Enables HTTP server functionality
- Use `style` for formatting-only changes (no code semantics). Prefer squashing style changes into the related code commit rather than separate noisy commits
- Mark breaking changes with `!` after the type (e.g., `refactor(api)!: ...`) and include migration notes; or add a `BREAKING CHANGE:` footer

Examples:
- `refactor(cuda): tighten unsafe surface across FFI`
- `test(attn): add last-token parity tests for odd head_dim`
- `style: format Rust code with rustfmt (no functional changes)`
- `refactor(api)!: remove deprecated d_data_base; use CudaContext handles`

We provide a commit template in `.gitmessage`. Enable it locally with:

```
git config commit.template .gitmessage
```

## Branching and PRs

- Prefer topic branches; avoid pushing directly to main
- Keep PRs focused and reviewable
- Update existing PRs instead of opening duplicates

## Features and Flags

- Keep non-CUDA path green: builds and tests should pass without CUDA
- Gate server behind the `server` feature
- CUDA code behind the `cuda` feature
- FP16 storage / FP32 compute; target `sm_52` for Tesla M40

## Code Style

- Rust: idiomatic, minimal comments, imports at top
- C/CUDA: `m40llm_*` naming for exported symbols; stable FFI
- Avoid redundant files; modify in place

## Testing

- Add CUDA-gated smoke tests for FFI when possible
- Do not add tests for docs or configuration-only changes

## Version Control Hygiene

- Do not commit large binaries or secrets
- Respect `.gitignore`
- Use `Co-authored-by` trailers when appropriate

## Git hooks (formatting and Conventional Commits)

We keep Git hooks in `.githooks`. To enable them locally, run:

```
git config core.hooksPath .githooks
```

Hooks provided:

## Two-phase commit flow (nonce-protected)

We use a hook-centric, two-step commit process to keep messages aligned with diffs and prevent accidental commits.

Enable hooks once (if not already):

```
git config core.hooksPath .githooks
```

Workflow:
- Draft: stage changes and run `git commit` (with -m/-F or editor). The commit-msg hook will:
  - Validate Conventional Commits via `cog verify`
  - Print the staged diff and your commit message
  - Compute a deterministic NONCE for the pair (staged diff, message)
  - Save a state lock at `.git/m40llm_nonce_state` and abort
- Finalize: if the diff and message are correct, run:

```
scripts/finalize-commit <NONCE>
```

The finalize step reuses `.git/COMMIT_EDITMSG` and validates that branch, staged index, message hash, and NONCE match the preview.

Recovery / tips:
- Your last drafted message is in `.git/COMMIT_EDITMSG`; a backup is saved by a `prepare-commit-msg` hook to `.git/COMMIT_EDITMSG.bak`.
- If you change staged files or the message, just run `git commit` again to get a new NONCE.
- If the state file seems stale, remove `.git/m40llm_nonce_state` and draft again.

- pre-commit: runs `cargo fmt --all -- --check` and rejects unformatted code
- commit-msg: enforces Conventional Commits via cocogitto and some hygiene checks
  - install cocogitto once: `cargo install cocogitto`
  - optionally enable strict history checks with: `export COG_STRICT=1`

We also provide a commit message template. Enable it with:

```
git config commit.template .gitmessage
```

