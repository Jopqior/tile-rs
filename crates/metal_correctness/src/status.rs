//! Per-case and run-level status. Fail and unverified are both non-success.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
    Unverified,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Fail => "fail",
            Status::Unverified => "unverified",
        }
    }

    pub fn is_success(self) -> bool {
        matches!(self, Status::Pass)
    }

    pub fn combine(self, other: Status) -> Status {
        match (self, other) {
            (Status::Fail, _) | (_, Status::Fail) => Status::Fail,
            (Status::Unverified, _) | (_, Status::Unverified) => Status::Unverified,
            (Status::Pass, Status::Pass) => Status::Pass,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CaseResult {
    pub id: String,
    pub status: Status,
    pub stage: String,
    pub detail: String,
    pub skipped_reason: Option<String>,
}

impl CaseResult {
    pub fn pass(id: &str) -> Self {
        Self {
            id: id.to_string(),
            status: Status::Pass,
            stage: "compare".into(),
            detail: "ok".into(),
            skipped_reason: None,
        }
    }

    pub fn fail(id: &str, stage: &str, detail: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            status: Status::Fail,
            stage: stage.into(),
            detail: detail.into(),
            skipped_reason: None,
        }
    }

    pub fn unverified(id: &str, stage: &str, detail: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            status: Status::Unverified,
            stage: stage.into(),
            detail: detail.into(),
            skipped_reason: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunSummary {
    pub results: Vec<CaseResult>,
    pub expected: Vec<String>,
    pub complete_list: bool,
}

impl RunSummary {
    pub fn overall(&self) -> Status {
        if !self.complete_list {
            // Single-case debug runs are never CI success evidence.
            let mut s = Status::Pass;
            for r in &self.results {
                s = s.combine(r.status);
            }
            return s;
        }
        let mut s = Status::Pass;
        for id in &self.expected {
            match self.results.iter().find(|r| r.id == *id) {
                None => s = s.combine(Status::Fail),
                Some(r) => s = s.combine(r.status),
            }
        }
        s
    }

    pub fn missing(&self) -> Vec<String> {
        self.expected
            .iter()
            .filter(|id| !self.results.iter().any(|r| r.id == **id))
            .cloned()
            .collect()
    }

    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let mut pass = 0;
        let mut fail = 0;
        let mut unverified = 0;
        for r in &self.results {
            match r.status {
                Status::Pass => pass += 1,
                Status::Fail => fail += 1,
                Status::Unverified => unverified += 1,
            }
        }
        (pass, fail, unverified, self.missing().len())
    }
}
