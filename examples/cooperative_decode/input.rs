use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub schema_version: u32,
    pub description: String,
    pub cpu_threads: usize,
    pub cpu_numa_node: usize,
    pub memory: Memory,
    pub evidence: BTreeMap<String, Evidence>,
    pub profiles: BTreeMap<String, Vec<Stage>>,
    pub requests: Vec<Request>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    pub host_capacity_bytes: u64,
    pub gpu_capacity_bytes: u64,
    /// Includes all weights, scratch, transfer buffers and retained copies.
    pub host_fixed_bytes: u64,
    pub gpu_fixed_bytes: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Evidence {
    Assumption {
        description: String,
    },
    /// Artifact integrity is checked by the CLI. Values remain declared inputs,
    /// not automatically extracted or independently validated measurements.
    Artifact {
        description: String,
        path: String,
        sha256: String,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    pub duration_ns: u64,
    pub evidence: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Assist {
    pub h2d: Cost,
    pub d2h: Cost,
    pub synchronization: Cost,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub name: String,
    pub cpu: Cost,
    pub gpu: Cost,
    /// Presence freezes this stage's routing in the assisted scenario.
    pub assist: Option<Assist>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: String,
    /// Request is already prefilled and all initial KV movement has finished.
    pub ready_ns: u64,
    pub is_cpu_resident: bool,
    pub host_peak_kv_bytes: u64,
    pub gpu_peak_kv_bytes: u64,
    /// One entry per generated-token position; costs may grow with context.
    pub token_profiles: Vec<String>,
}

impl Trace {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "unsupported schema");
        ensure!(
            !self.description.trim().is_empty(),
            "missing trace description"
        );
        ensure!(
            (1..=256).contains(&self.cpu_threads),
            "invalid CPU worker count"
        );
        ensure!(
            (1..=1024).contains(&self.requests.len()),
            "invalid request count"
        );
        ensure!(
            (1..=4096).contains(&self.profiles.len()),
            "invalid profile count"
        );
        ensure!(
            self.memory.host_capacity_bytes > 0 && self.memory.gpu_capacity_bytes > 0,
            "memory capacities must be positive"
        );
        for evidence in self.evidence.values() {
            let description = match evidence {
                Evidence::Assumption { description } => description,
                Evidence::Artifact {
                    description,
                    path,
                    sha256,
                } => {
                    ensure!(!path.is_empty(), "empty artifact path");
                    ensure!(
                        sha256.len() == 64 && sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                        "invalid artifact hash"
                    );
                    description
                }
            };
            ensure!(!description.trim().is_empty(), "empty cost evidence");
        }
        for (name, stages) in &self.profiles {
            ensure!(
                !name.is_empty() && (1..=512).contains(&stages.len()),
                "invalid profile"
            );
            let mut names = BTreeSet::new();
            for stage in stages {
                ensure!(
                    !stage.name.is_empty() && names.insert(&stage.name),
                    "duplicate/empty stage"
                );
                self.validate_cost(&stage.cpu)?;
                self.validate_cost(&stage.gpu)?;
                if let Some(assist) = &stage.assist {
                    self.validate_cost(&assist.h2d)?;
                    self.validate_cost(&assist.d2h)?;
                    self.validate_cost(&assist.synchronization)?;
                    ensure!(
                        assist.h2d_bytes > 0 && assist.d2h_bytes > 0,
                        "empty assist transfer"
                    );
                }
            }
        }
        let mut ids = BTreeSet::new();
        let mut step_count = 0usize;
        let mut duration_bound = self.requests.iter().map(|r| r.ready_ns).max().unwrap_or(0);
        let mut transfer_bound = 0u64;
        for request in &self.requests {
            ensure!(
                !request.id.is_empty() && ids.insert(&request.id),
                "duplicate/empty request ID"
            );
            ensure!(!request.token_profiles.is_empty(), "request has no tokens");
            ensure!(
                request.host_peak_kv_bytes > 0 && request.gpu_peak_kv_bytes > 0,
                "missing peak KV reservation"
            );
            for name in &request.token_profiles {
                let stages = self.profiles.get(name).context("missing token profile")?;
                for stage in stages {
                    step_count = step_count.checked_add(5).context("work count overflow")?;
                    let mut costs = vec![&stage.cpu, &stage.gpu];
                    if let Some(assist) = &stage.assist {
                        costs.extend([&assist.h2d, &assist.d2h, &assist.synchronization]);
                        transfer_bound = transfer_bound
                            .checked_add(assist.h2d_bytes)
                            .and_then(|v| v.checked_add(assist.d2h_bytes))
                            .context("transfer overflow")?;
                    }
                    for cost in costs {
                        duration_bound = duration_bound
                            .checked_add(cost.duration_ns)
                            .context("simulation clock overflow")?;
                    }
                }
            }
        }
        ensure!(step_count <= 200_000, "trace exceeds 200000-step bound");
        Ok(())
    }

    fn validate_cost(&self, cost: &Cost) -> Result<()> {
        ensure!(
            cost.duration_ns > 0,
            "cost must be positive integer nanoseconds"
        );
        ensure!(
            self.evidence.contains_key(&cost.evidence),
            "missing cost evidence"
        );
        Ok(())
    }
}
