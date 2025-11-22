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
