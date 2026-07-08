use std::fmt;

/// Convenience alias for fallible `sc-core` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by `sc-core`.
///
/// Intentionally minimal at M0; variants are added by the milestones that can
/// produce them. `#[non_exhaustive]` keeps adding variants a non-breaking
/// change for downstream matches.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Input data or parameters were invalid; the string explains why.
    InvalidInput(String),
    /// A serialized model could not be loaded: bad magic, an unsupported format
    /// version, a corrupt/truncated payload, or an internally inconsistent one.
    /// The string explains why. Produced only by the `persist` feature's
    /// `from_bytes`; loading always fails with this error rather than panicking.
    Deserialize(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::Deserialize(msg) => write!(f, "deserialize: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
