#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionBackendMode {
    Auto,
    FastFits,
    LargeModel,
}

impl ProjectionBackendMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "fast-fits" | "fast_fits" => Some(Self::FastFits),
            "large-model" | "large_model" => Some(Self::LargeModel),
            _ => None,
        }
    }

    pub fn from_env() -> Self {
        std::env::var("M40LLM_PROJECTION_BACKEND")
            .ok()
            .and_then(|value| Self::parse(value.trim()))
            .unwrap_or(Self::Auto)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionBackend {
    FastFits,
    LargeModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionBackendEstimate {
    pub weights_bytes: usize,
    pub materialized_f32_bytes: usize,
    pub workspace_bytes: usize,
    pub kv_bytes: usize,
}

impl ProjectionBackendEstimate {
    pub fn total_with_materialization(&self) -> Option<usize> {
        self.weights_bytes
            .checked_add(self.materialized_f32_bytes)?
            .checked_add(self.workspace_bytes)?
            .checked_add(self.kv_bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionBackendDecision {
    pub requested: ProjectionBackendMode,
    pub selected: ProjectionBackend,
    pub estimate: ProjectionBackendEstimate,
    pub budget_bytes: Option<usize>,
    pub cublas_available: bool,
    pub materialization_enabled: bool,
    pub reason: &'static str,
}

impl ProjectionBackendDecision {
    pub fn allows_materialized_f32(&self) -> bool {
        self.selected == ProjectionBackend::FastFits
            && self.cublas_available
            && self.materialization_enabled
    }
}

pub fn choose_projection_backend(
    requested: ProjectionBackendMode,
    estimate: ProjectionBackendEstimate,
    budget_bytes: Option<usize>,
    cublas_available: bool,
    materialization_enabled: bool,
) -> ProjectionBackendDecision {
    let total = estimate.total_with_materialization();
    let (selected, reason) = match requested {
        ProjectionBackendMode::FastFits => {
            if !materialization_enabled {
                (
                    ProjectionBackend::LargeModel,
                    "fast-fits requested but materialized FP32 weights disabled",
                )
            } else if !cublas_available {
                (
                    ProjectionBackend::LargeModel,
                    "fast-fits requested but cuBLAS unavailable",
                )
            } else {
                (ProjectionBackend::FastFits, "fast-fits requested")
            }
        }
        ProjectionBackendMode::LargeModel => {
            (ProjectionBackend::LargeModel, "large-model requested")
        }
        ProjectionBackendMode::Auto => {
            if !materialization_enabled {
                (
                    ProjectionBackend::LargeModel,
                    "materialized FP32 weights disabled",
                )
            } else if !cublas_available {
                (ProjectionBackend::LargeModel, "cuBLAS unavailable")
            } else if let (Some(total), Some(budget)) = (total, budget_bytes) {
                if total <= budget {
                    (
                        ProjectionBackend::FastFits,
                        "estimated fast-fits total within budget",
                    )
                } else {
                    (
                        ProjectionBackend::LargeModel,
                        "estimated fast-fits total exceeds budget",
                    )
                }
            } else {
                (
                    ProjectionBackend::FastFits,
                    "no fast-fits budget configured",
                )
            }
        }
    };

    ProjectionBackendDecision {
        requested,
        selected,
        estimate,
        budget_bytes,
        cublas_available,
        materialization_enabled,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(total_materialized: usize) -> ProjectionBackendEstimate {
        ProjectionBackendEstimate {
            weights_bytes: 100,
            materialized_f32_bytes: total_materialized,
            workspace_bytes: 10,
            kv_bytes: 20,
        }
    }

    #[test]
    fn auto_selects_fast_fits_within_budget() {
        let decision = choose_projection_backend(
            ProjectionBackendMode::Auto,
            estimate(50),
            Some(200),
            true,
            true,
        );
        assert_eq!(decision.selected, ProjectionBackend::FastFits);
        assert!(decision.allows_materialized_f32());
    }

    #[test]
    fn auto_selects_large_model_when_over_budget() {
        let decision = choose_projection_backend(
            ProjectionBackendMode::Auto,
            estimate(500),
            Some(200),
            true,
            true,
        );
        assert_eq!(decision.selected, ProjectionBackend::LargeModel);
        assert!(!decision.allows_materialized_f32());
    }

    #[test]
    fn explicit_large_model_disables_materialization() {
        let decision = choose_projection_backend(
            ProjectionBackendMode::LargeModel,
            estimate(50),
            None,
            true,
            true,
        );
        assert_eq!(decision.selected, ProjectionBackend::LargeModel);
        assert!(!decision.allows_materialized_f32());
    }

    #[test]
    fn materialization_env_disable_overrides_fast_fits_request() {
        let decision = choose_projection_backend(
            ProjectionBackendMode::FastFits,
            estimate(50),
            None,
            true,
            false,
        );
        assert_eq!(decision.selected, ProjectionBackend::LargeModel);
        assert!(!decision.allows_materialized_f32());
    }
}
