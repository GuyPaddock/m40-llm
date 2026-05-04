use super::meta::dtype_size_bytes;
use super::DeviceTensorView;
use crate::gguf::{GgmlDType, GgufTensor};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::ffi::c_void;

fn contiguous_strides(shape: &[u64]) -> Result<Vec<usize>> {
    if shape.is_empty() {
        anyhow::bail!("tensor shape must be non-empty");
    }
    let mut strides_rev = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for (dim_idx, &dim) in shape.iter().enumerate().rev() {
        if dim == 0 {
            anyhow::bail!("tensor dim {dim_idx} must be > 0");
        }
        strides_rev.push(stride);
        let dim_usize =
            usize::try_from(dim).with_context(|| format!("dim {dim_idx} does not fit in usize"))?;
        stride = stride
            .checked_mul(dim_usize)
            .context("stride overflow while building tensor view")?;
    }
    strides_rev.reverse();
    Ok(strides_rev)
}

#[allow(clippy::collapsible_if)]
pub(super) fn build_device_tensor_views(
    tensors: &[GgufTensor],
    #[allow(unused_variables)] d_base: *mut c_void,
    weights_len: usize,
) -> Result<HashMap<String, DeviceTensorView>> {
    let mut map = HashMap::with_capacity(tensors.len());
    for t in tensors {
        if t.shape.is_empty() {
            anyhow::bail!("tensor '{}' has empty shape", t.name);
        }
        if let Some((idx, _)) = t.shape.iter().enumerate().find(|(_, &d)| d == 0) {
            anyhow::bail!("tensor '{}' has zero in dimension {}", t.name, idx);
        }
        // Compute size and perform bounds/alignment checks
        let offset_usize: usize = t
            .offset
            .try_into()
            .context("tensor offset does not fit in usize")?;
        // Alignment by dtype
        let align = match t.dtype {
            GgmlDType::F16 => 2usize,
            GgmlDType::F32 => 4usize,
            _ => 1usize,
        };
        if align > 1 && !offset_usize.is_multiple_of(align) {
            anyhow::bail!(
                "tensor '{}' offset {} misaligned for {:?} (align {})",
                t.name,
                offset_usize,
                t.dtype,
                align
            );
        }
        let layout = dtype_size_bytes(t.dtype)
            .with_context(|| format!("tensor '{}' unsupported dtype: {:?}", t.name, t.dtype))?;
        // Known element size: check shape product and bounds within weights_len
        let n_elems_u64: u64 = t.shape.iter().copied().product::<u64>();
        let n_elems: usize =
            usize::try_from(n_elems_u64).context("tensor element count does not fit in usize")?;
        let n_blocks = (n_elems + layout.block_elems - 1) / layout.block_elems;
        let need = n_blocks
            .checked_mul(layout.block_bytes)
            .context("tensor size overflow")?;
        let end = offset_usize
            .checked_add(need)
            .context("tensor end offset overflow")?;
        let nbytes = need;
        let end_ok = end <= weights_len;
        if !end_ok {
            anyhow::bail!(
                "tensor '{}' overflows weights blob or starts beyond end (off={}, nbytes={}, total={})",
                t.name, offset_usize, nbytes, weights_len
            );
        }
        let strides = contiguous_strides(&t.shape)
            .with_context(|| format!("tensor '{}' stride computation failed", t.name))?;
        // Safe device pointer arithmetic
        #[cfg(feature = "cuda")]
        let dptr: *mut c_void = {
            if d_base.is_null() {
                std::ptr::null_mut()
            } else {
                let base = d_base as usize;
                let addr = base
                    .checked_add(offset_usize)
                    .context("device pointer offset overflow")?;
                let ptr = addr as *mut c_void;
                if std::env::var("M40LLM_TENSOR_VIEW_LOG").ok().as_deref() == Some("1") {
                    eprintln!(
                        "[cuda] tensor_view: name={}, base={:?}, offset={}, ptr={:?}",
                        t.name, d_base, offset_usize, ptr
                    );
                }
                ptr
            }
        };
        #[cfg(not(feature = "cuda"))]
        let _dptr: *mut c_void = std::ptr::null_mut();
        let view = DeviceTensorView {
            dtype: t.dtype,
            shape: t.shape.clone(),
            strides,
            byte_offset: t.offset,
            nbytes,
            #[cfg(feature = "cuda")]
            dptr,
        };
        map.insert(t.name.clone(), view);
    }
    Ok(map)
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tensor_view_tests {
    use super::*;

    #[test]
    fn device_pointer_matches_base_and_offset() {
        let tensors = vec![GgufTensor {
            name: "w".to_string(),
            dtype: GgmlDType::F16,
            shape: vec![2, 2],
            offset: 8,
        }];
        let base = 0x1000usize as *mut c_void;
        let views = build_device_tensor_views(&tensors, base, 64).expect("views");
        let w = views.get("w").expect("tensor view");
        assert_eq!(w.dptr as usize, base as usize + 8);
    }

    #[test]
    fn null_device_base_keeps_tensor_ptr_null() {
        let tensors = vec![GgufTensor {
            name: "w".to_string(),
            dtype: GgmlDType::F16,
            shape: vec![1, 1],
            offset: 0,
        }];
        let views = build_device_tensor_views(&tensors, std::ptr::null_mut(), 16).expect("views");
        let w = views.get("w").expect("tensor view");
        assert!(w.dptr.is_null());
    }
}
