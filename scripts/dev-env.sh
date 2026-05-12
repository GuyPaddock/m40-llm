#!/usr/bin/env bash
# Project-local tooling environment for agents and hooks.
#
# Source this before running commands when the host toolchain is incomplete:
#   source scripts/dev-env.sh

if [ -z "${MAMBA_ROOT_PREFIX:-}" ] || [ ! -d "$MAMBA_ROOT_PREFIX/envs/m40-rustfmt" ]; then
  export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
fi

M40_RUSTFMT_BIN="$MAMBA_ROOT_PREFIX/envs/m40-rustfmt/bin"
M40_RUSTFMT_SYSROOT="$MAMBA_ROOT_PREFIX/envs/m40-rustfmt/x86_64-conda-linux-gnu/sysroot"
if [ -d "$M40_RUSTFMT_SYSROOT/usr/include" ]; then
  export M40LLM_SYSROOT="${M40LLM_SYSROOT:-$M40_RUSTFMT_SYSROOT}"
fi

M40_LOCAL_BIN="$HOME/.local/bin"
if [ -d "$M40_LOCAL_BIN" ]; then
  case ":$PATH:" in
    *":$M40_LOCAL_BIN:"*) ;;
    *) export PATH="$M40_LOCAL_BIN:$PATH" ;;
  esac
fi

M40_CARGO_BIN="$HOME/.cargo/bin"
if [ -d "$M40_CARGO_BIN" ]; then
  case ":$PATH:" in
    *":$M40_CARGO_BIN:"*) ;;
    *) export PATH="$M40_CARGO_BIN:$PATH" ;;
  esac
fi

if [ -d "$M40_RUSTFMT_BIN" ]; then
  case ":$PATH:" in
    *":$M40_RUSTFMT_BIN:"*) ;;
    *) export PATH="$M40_RUSTFMT_BIN:$PATH" ;;
  esac
fi
