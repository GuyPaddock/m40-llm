User:
Okay, time for some next steps!

> cleanly factor out a DeviceAllocator API in Rust
---
Agent:
Absolutely — now that `m40-llm` has a stable CUDA/Rust FFI layer, the next major cleanup is to **abstract device memory management**.

Right now:

* Each module directly calls `cuda.device_malloc` / `cuda.device_free`
* Scratch buffers (`d_q`, `d_k`, etc) are manually allocated per model instance
* There’s no tracking of allocations → no reuse → no pooling → no debugging visibility
* We can’t easily plug in:

    * stream-ordered allocators
    * caching allocators
    * segmented arenas
    * page-locked host staging buffers
    * memory introspection tools

The solution is a **centralized DeviceAllocator API** that:

✔ Lives in Rust
✔ Owns all CUDA device memory
✔ Exposes typed allocation helpers (e.g., `alloc_f16`, `alloc_f32`)
✔ Automatically frees memory on drop
✔ Is pluggable (pooling backends, arenas, per-stream allocators)
✔ Plays well with `Arc<CudaContext>`

This is very similar to:

* PyTorch’s CUDACachingAllocator
* TensorRT’s device memory pools
* CUDA Memory Pools (cudaMallocAsync)

But built in Rust for our minimal runtime.

---

# 🚀 Goal: `DeviceAllocator`

A clean Rust API:

```rust
let alloc = DeviceAllocator::new(cuda.clone());

let buf1: DevicePtr<f32> = alloc.alloc_f32(4096)?;
let buf2: DevicePtr<half> = alloc.alloc_f16(8192)?;

alloc.free(buf1)?;
alloc.free(buf2)?;
```

Or even more idiomatic:

```rust
let tmp = alloc.scope(|scope| {
    let q = scope.alloc_f16(d_model)?;
    let k = scope.alloc_f16(d_model)?;
    let v = scope.alloc_f16(d_model)?;

    run_qkv(q, k, v)?;
    Ok(())
});
```

---

# 🎯 Feature Checklist

| Feature                                | Supported? | Notes                        |
| -------------------------------------- | ---------- | ---------------------------- |
| RAII automatic free                    | ✔          | Drop frees memory            |
| Typed device pointers                  | ✔          | `DevicePtr<T>`               |
| Error-safe                             | ✔          | Uses `anyhow::Result`        |
| Can wrap CUDA stream-ordered alloc API | Future     | Works with `cudaMallocAsync` |
| Reusable pools                         | Future     | Caching allocator            |
| Segment/arena-based allocation         | Future     | For large model buffers      |
| Zero-copy host pinned memory           | Future     | `cudaHostAlloc`              |

---

# 🧱 Implementation

## 1. New module: `src/device_allocator.rs`

### First: The typed device pointer

```rust
use std::ffi::c_void;
use std::marker::PhantomData;

pub struct DevicePtr<T> {
    pub raw: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> DevicePtr<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.raw
    }
}
```

### Allocation backend enum

We support:

* `Basic` → uses `cudaMalloc` / `cudaFree`
* Later: `AsyncPool` → CUDA async memory pools
* Later: `BumpArena` → global scratch arena per model

```rust
#[derive(Clone, Copy)]
pub enum AllocBackend {
    Basic,
    // Future:
    // AsyncPool,
    // Arena,
}
```

### The allocator struct

```rust
use crate::cuda::CudaContext;
use anyhow::{Result, anyhow};

pub struct DeviceAllocator {
    cuda: std::sync::Arc<CudaContext>,
    backend: AllocBackend,
}
```

Constructor:

```rust
impl DeviceAllocator {
    pub fn new(cuda: std::sync::Arc<CudaContext>) -> Self {
        Self {
            cuda,
            backend: AllocBackend::Basic,
        }
    }

    pub fn with_backend(cuda: std::sync::Arc<CudaContext>, backend: AllocBackend) -> Self {
        Self { cuda, backend }
    }
}
```

---

## 2. Allocation functions

### Raw allocator

```rust
impl DeviceAllocator {
    fn alloc_raw(&self, n_bytes: usize) -> Result<*mut c_void> {
        match self.backend {
            AllocBackend::Basic => self.cuda.device_malloc(n_bytes),
        }
    }

    fn free_raw(&self, ptr: *mut c_void) -> Result<()> {
        match self.backend {
            AllocBackend::Basic => self.cuda.device_free(ptr),
        }
    }
}
```

---

## 3. Typed allocators

### FP16 allocation

```rust
use half::f16;

impl DeviceAllocator {
    pub fn alloc_f16(&self, n: usize) -> Result<DevicePtr<f16>> {
        let bytes = n * 2;
        let raw = self.alloc_raw(bytes)?;
        Ok(DevicePtr {
            raw,
            len: n,
            _marker: std::marker::PhantomData,
        })
    }
}
```

### FP32 allocation

```rust
impl DeviceAllocator {
    pub fn alloc_f32(&self, n: usize) -> Result<DevicePtr<f32>> {
        let bytes = n * 4;
        let raw = self.alloc_raw(bytes)?;
        Ok(DevicePtr {
            raw,
            len: n,
            _marker: std::marker::PhantomData,
        })
    }
}
```

### Arbitrary-typed allocation

```rust
impl DeviceAllocator {
    pub fn alloc_type<T>(&self, n: usize) -> Result<DevicePtr<T>> {
        let bytes = n * std::mem::size_of::<T>();
        let raw = self.alloc_raw(bytes)?;
        Ok(DevicePtr {
            raw,
            len: n,
            _marker: std::marker::PhantomData,
        })
    }
}
```

---

## 4. Free via RAII (Drop)

```rust
impl<T> Drop for DevicePtr<T> {
    fn drop(&mut self) {
        // This MUST NOT panic.
        if !self.raw.is_null() {
            // We need the allocator to free — use a drop guard later.
        }
    }
}
```

We *can’t* free memory here until we track which allocator created this pointer.

Solution: attach a pointer to allocator inside `DevicePtr`.

Let’s fix that:

```rust
pub struct DevicePtr<T> {
    pub raw: *mut c_void,
    len: usize,
    alloc: std::sync::Arc<DeviceAllocator>,
    _marker: PhantomData<T>,
}
```

And in the allocator:

```rust
DevicePtr {
    raw,
    len: n,
    alloc: self.clone(),
    _marker: PhantomData,
}
```

Now RAII works:

```rust
impl<T> Drop for DevicePtr<T> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // ignore errors — cannot panic in Drop
            let _ = self.alloc.free_raw(self.raw);
        }
    }
}
```

---

## 5. Scoped temporary allocator (for scratch Q/K/V buffers)

### Example usage:

```rust
alloc.scope(|scope| {
    let q = scope.alloc_f16(d_model)?;
    let k = scope.alloc_f16(d_model)?;
    let v = scope.alloc_f16(d_model)?;

    // use q,k,v for this block…

    Ok(())
});
```

Implementation:

```rust
pub struct TempScope<'a> {
    alloc: &'a DeviceAllocator,
    ptrs: Vec<*mut c_void>,
}

impl DeviceAllocator {
    pub fn scope<R>(&self, f: impl FnOnce(&mut TempScope) -> Result<R>) -> Result<R> {
        let mut scope = TempScope {
            alloc: self,
            ptrs: Vec::new(),
        };
        let out = f(&mut scope);
        scope.cleanup()?;
        out
    }
}

impl<'a> TempScope<'a> {
    pub fn alloc_f16(&mut self, n: usize) -> Result<DevicePtr<f16>> {
        let bytes = n * 2;
        let raw = self.alloc.alloc_raw(bytes)?;
        self.ptrs.push(raw);
        Ok(DevicePtr {
            raw,
            len: n,
            alloc: std::sync::Arc::new(self.alloc.clone()),
            _marker: PhantomData,
        })
    }

    fn cleanup(&mut self) -> Result<()> {
        for &ptr in &self.ptrs {
            self.alloc.free_raw(ptr)?;
        }
        self.ptrs.clear();
        Ok(())
    }
}
```

Now all temporary buffers go away automatically.

---

# 🏗 Integrating Into `m40-llm`

You will add:

```
src/device_allocator.rs
```

And in `src/lib.rs` or wherever:

```rust
pub mod device_allocator;
```

In `model.rs`, replace raw CUDA allocations:

```rust
let d_q    = cuda.device_malloc(qkv_len * 2)?;
let d_k    = cuda.device_malloc(qkv_len * 2)?;
let d_v    = cuda.device_malloc(qkv_len * 2)?;
```

with:

```rust
let alloc = DeviceAllocator::new(cuda.clone());

let d_q = alloc.alloc_f16(qkv_len)?;
let d_k = alloc.alloc_f16(qkv_len)?;
let d_v = alloc.alloc_f16(qkv_len)?;
```

Scratch spaces become typed, leak-free, auto-freed.

KV cache remains allocated through its own FFI lifecycle — that’s correct.

---

# 🎉 WHAT YOU GET NOW

After adding this module:

✔ All temporary buffers live in **one unified allocation system**
✔ Every `DevicePtr<T>` is cleared automatically
✔ No manual `device_free()` calls
✔ Typed device memory (f16, f32, arbitrary)
✔ Ready for future optimizations:

* `cudaMallocAsync`
* suballocated arenas
* caching allocator
* per-stream pools
* registered host memory

---

Next steps:
A. Add a **caching allocator** (PyTorch-style)
B. Add **cudaMallocAsync + CUDA memory pools** backend
C. Add a **persistent decode scratch pool**
D. Convert the model loader to use the new allocator
E. Implement **MLP + RMSNorm kernels**