//! Weight-only BCQ fit adapted from ggml's sa_fit_group and sa_quant_row.
//! Copyright (c) 2023-2026 The ggml authors. See LICENSE.txt.
//! Offline FP64 arithmetic intentionally differs from upstream FP32 fitting.

use super::{GROUP_WEIGHTS, SCALE_OFFSET, SIGN_PLANES};
use anyhow::{ensure, Result};
use half::f16;

pub(super) fn encode_block(source: &[f32], destination: &mut [u8]) -> Result<()> {
    let amplitudes = [
        fit_group(&source[..GROUP_WEIGHTS]),
        fit_group(&source[GROUP_WEIGHTS..]),
    ];
    let maximum = amplitudes.iter().flatten().copied().fold(0.0_f64, f64::max);
    let scale = f16::from_f64(maximum / 255.0);
    ensure!(
        scale.is_finite(),
        "fitted scale exceeds finite FP16 storage"
    );
    destination.fill(0);
    destination[SCALE_OFFSET..].copy_from_slice(&scale.to_le_bytes());
    let scale = scale.to_f64();
    for (group, amplitudes) in amplitudes.iter().enumerate() {
        let mut quantized = [0.0; SIGN_PLANES];
        for (plane, &amplitude) in amplitudes.iter().enumerate() {
            let code = if scale > 0.0 {
                (amplitude / scale).round_ties_even().clamp(0.0, 255.0) as u8
            } else {
                0
            };
            destination[128 + group * SIGN_PLANES + plane] = code;
            quantized[plane] = scale * f64::from(code);
        }
        let levels = calculate_levels(&quantized);
        for index in group * GROUP_WEIGHTS..(group + 1) * GROUP_WEIGHTS {
            let code = select_nearest(f64::from(source[index]), &levels);
            for plane in 0..SIGN_PLANES {
                if code & (1 << plane) != 0 {
                    destination[plane * 32 + index / 8] |= 1 << (index % 8);
                }
            }
        }
    }
    Ok(())
}

fn fit_group(source: &[f32]) -> [f64; SIGN_PLANES] {
    let mut residual = [0.0; GROUP_WEIGHTS];
    for (destination, &value) in residual.iter_mut().zip(source) {
        *destination = f64::from(value);
    }
    let mut amplitudes = [0.0; SIGN_PLANES];
    for amplitude in &mut amplitudes {
        *amplitude = residual.iter().map(|value| value.abs()).sum::<f64>() / GROUP_WEIGHTS as f64;
        for value in &mut residual {
            *value -= if *value >= 0.0 {
                *amplitude
            } else {
                -*amplitude
            };
        }
    }
    for _ in 0..4 {
        let levels = calculate_levels(&amplitudes);
        let mut system = [[0.0; SIGN_PLANES + 1]; SIGN_PLANES];
        for &value in source {
            let value = f64::from(value);
            let code = select_nearest(value, &levels);
            let signs: [f64; SIGN_PLANES] =
                std::array::from_fn(|plane| if code & (1 << plane) != 0 { 1.0 } else { -1.0 });
            for (&sign, equation) in signs.iter().zip(&mut system) {
                for (column, &other_sign) in signs.iter().enumerate() {
                    equation[column] += sign * other_sign;
                }
                equation[SIGN_PLANES] += sign * value;
            }
        }
        let Some(mut fitted) = solve_system(system) else {
            break;
        };
        fitted.sort_by(|a, b| b.total_cmp(a));
        amplitudes = fitted;
    }
    amplitudes
}

fn calculate_levels(amplitudes: &[f64; SIGN_PLANES]) -> [f64; 16] {
    std::array::from_fn(|code| {
        amplitudes
            .iter()
            .enumerate()
            .map(|(plane, &amplitude)| {
                if code & (1 << plane) != 0 {
                    amplitude
                } else {
                    -amplitude
                }
            })
            .sum()
    })
}

fn select_nearest(value: f64, levels: &[f64; 16]) -> usize {
    let mut selected = 0;
    let mut distance = f64::INFINITY;
    for (code, &level) in levels.iter().enumerate() {
        let candidate = (value - level).abs();
        if candidate < distance {
            distance = candidate;
            selected = code;
        }
    }
    selected
}

// Row/column indexing keeps the pivoted elimination order explicit.
#[allow(clippy::needless_range_loop)]
fn solve_system(mut system: [[f64; SIGN_PLANES + 1]; SIGN_PLANES]) -> Option<[f64; SIGN_PLANES]> {
    for pivot in 0..SIGN_PLANES {
        let mut selected = pivot;
        for row in pivot + 1..SIGN_PLANES {
            if system[row][pivot].abs() > system[selected][pivot].abs() {
                selected = row;
            }
        }
        if system[selected][pivot].abs() < 1e-20 {
            return None;
        }
        system.swap(pivot, selected);
        for row in 0..SIGN_PLANES {
            if row == pivot {
                continue;
            }
            let factor = system[row][pivot] / system[pivot][pivot];
            for column in pivot..=SIGN_PLANES {
                system[row][column] -= factor * system[pivot][column];
            }
        }
    }
    Some(std::array::from_fn(|row| {
        (system[row][SIGN_PLANES] / system[row][row]).abs()
    }))
}
