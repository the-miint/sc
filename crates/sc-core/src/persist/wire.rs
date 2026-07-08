//! The frozen on-disk format, isolated here.
//!
//! These are the *only* serde-deriving types in `sc-core`. They mirror the core
//! model/param types so the wire format is decoupled from internal struct layout:
//! the core types carry no serde derives, and postcard (which is
//! non-self-describing — field order *is* the format) sees only these mirrors.
//! Format evolution is therefore a localized change to this file (design §9).
//!
//! Any reordering/retyping of a field here changes the bytes; the committed
//! golden fixture (`tests/persist.rs`) is the tripwire that catches it, and the
//! envelope's format version (in [`super`]) gates loadability.

use serde::{Deserialize, Serialize};

use crate::{Criterion, MaxFeatures, MaxSamples, MinSamples, RandomForestParams, TreeParams};

/// Which estimator the payload holds (redundant with `Task` but stored
/// explicitly per design, and cheap to inspect without decoding the trees).
#[derive(Serialize, Deserialize)]
pub(crate) enum KindWire {
    Classifier,
    Regressor,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum CriterionWire {
    Gini,
    Entropy,
    SquaredError,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum MaxFeaturesWire {
    All,
    Sqrt,
    Log2,
    Fraction(f64),
    Count(usize),
}

#[derive(Serialize, Deserialize)]
pub(crate) enum MinSamplesWire {
    Count(usize),
    Fraction(f64),
}

#[derive(Serialize, Deserialize)]
pub(crate) enum MaxSamplesWire {
    All,
    Count(usize),
    Fraction(f64),
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TreeParamsWire {
    pub criterion: CriterionWire,
    pub max_depth: Option<usize>,
    pub min_samples_split: MinSamplesWire,
    pub min_samples_leaf: MinSamplesWire,
    pub min_weight_fraction_leaf: f64,
    pub max_features: MaxFeaturesWire,
    pub min_impurity_decrease: f64,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ParamsWire {
    pub tree: TreeParamsWire,
    pub n_estimators: usize,
    pub bootstrap: bool,
    pub max_samples: MaxSamplesWire,
}

/// One tree's struct-of-arrays (compact native widths). `value` is flattened
/// `n_nodes * value_stride`; `weighted_n_node_samples` is the per-node cover kept
/// so a reloaded model supports importances and (M10) TreeSHAP. `n_nodes`,
/// `value_stride`, and the tree's `Task` are all derived on load, so not stored.
#[derive(Serialize, Deserialize)]
pub(crate) struct TreeWire {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f64>,
    pub value: Vec<f64>,
    pub impurity: Vec<f64>,
    pub n_node_samples: Vec<u64>,
    pub weighted_n_node_samples: Vec<f64>,
}

/// The postcard payload — everything after the raw magic + version header.
#[derive(Serialize, Deserialize)]
pub(crate) struct Payload {
    /// Informational only (our semver at save time); never gates loadability.
    pub lib_semver: String,
    pub kind: KindWire,
    pub n_features: usize,
    /// `Some` iff classifier (probability-column labels, in `classes_` order).
    pub classes: Option<Vec<String>>,
    /// Hyperparameters used to fit the model (optional).
    pub params: Option<ParamsWire>,
    /// Optional feature names aligned to the model's columns.
    pub feature_ids: Option<Vec<String>>,
    /// RFECV-selected feature set — reserved for M6; `None` for an M5 model.
    pub rfe_selection: Option<Vec<usize>>,
    pub trees: Vec<TreeWire>,
}

// --- conversions: core -> wire (save) and wire -> core (load) --------------- //
// Params carry no construction invariants (validation is a separate step), so
// these are purely structural. The Copy value-enums convert by value.

impl From<Criterion> for CriterionWire {
    fn from(c: Criterion) -> Self {
        match c {
            Criterion::Gini => CriterionWire::Gini,
            Criterion::Entropy => CriterionWire::Entropy,
            Criterion::SquaredError => CriterionWire::SquaredError,
        }
    }
}
impl From<CriterionWire> for Criterion {
    fn from(w: CriterionWire) -> Self {
        match w {
            CriterionWire::Gini => Criterion::Gini,
            CriterionWire::Entropy => Criterion::Entropy,
            CriterionWire::SquaredError => Criterion::SquaredError,
        }
    }
}

impl From<MaxFeatures> for MaxFeaturesWire {
    fn from(m: MaxFeatures) -> Self {
        match m {
            MaxFeatures::All => MaxFeaturesWire::All,
            MaxFeatures::Sqrt => MaxFeaturesWire::Sqrt,
            MaxFeatures::Log2 => MaxFeaturesWire::Log2,
            MaxFeatures::Fraction(f) => MaxFeaturesWire::Fraction(f),
            MaxFeatures::Count(c) => MaxFeaturesWire::Count(c),
        }
    }
}
impl From<MaxFeaturesWire> for MaxFeatures {
    fn from(w: MaxFeaturesWire) -> Self {
        match w {
            MaxFeaturesWire::All => MaxFeatures::All,
            MaxFeaturesWire::Sqrt => MaxFeatures::Sqrt,
            MaxFeaturesWire::Log2 => MaxFeatures::Log2,
            MaxFeaturesWire::Fraction(f) => MaxFeatures::Fraction(f),
            MaxFeaturesWire::Count(c) => MaxFeatures::Count(c),
        }
    }
}

impl From<MinSamples> for MinSamplesWire {
    fn from(m: MinSamples) -> Self {
        match m {
            MinSamples::Count(c) => MinSamplesWire::Count(c),
            MinSamples::Fraction(f) => MinSamplesWire::Fraction(f),
        }
    }
}
impl From<MinSamplesWire> for MinSamples {
    fn from(w: MinSamplesWire) -> Self {
        match w {
            MinSamplesWire::Count(c) => MinSamples::Count(c),
            MinSamplesWire::Fraction(f) => MinSamples::Fraction(f),
        }
    }
}

impl From<MaxSamples> for MaxSamplesWire {
    fn from(m: MaxSamples) -> Self {
        match m {
            MaxSamples::All => MaxSamplesWire::All,
            MaxSamples::Count(c) => MaxSamplesWire::Count(c),
            MaxSamples::Fraction(f) => MaxSamplesWire::Fraction(f),
        }
    }
}
impl From<MaxSamplesWire> for MaxSamples {
    fn from(w: MaxSamplesWire) -> Self {
        match w {
            MaxSamplesWire::All => MaxSamples::All,
            MaxSamplesWire::Count(c) => MaxSamples::Count(c),
            MaxSamplesWire::Fraction(f) => MaxSamples::Fraction(f),
        }
    }
}

impl From<&TreeParams> for TreeParamsWire {
    fn from(p: &TreeParams) -> Self {
        TreeParamsWire {
            criterion: p.criterion.into(),
            max_depth: p.max_depth,
            min_samples_split: p.min_samples_split.into(),
            min_samples_leaf: p.min_samples_leaf.into(),
            min_weight_fraction_leaf: p.min_weight_fraction_leaf,
            max_features: p.max_features.into(),
            min_impurity_decrease: p.min_impurity_decrease,
        }
    }
}
impl From<TreeParamsWire> for TreeParams {
    fn from(w: TreeParamsWire) -> Self {
        TreeParams {
            criterion: w.criterion.into(),
            max_depth: w.max_depth,
            min_samples_split: w.min_samples_split.into(),
            min_samples_leaf: w.min_samples_leaf.into(),
            min_weight_fraction_leaf: w.min_weight_fraction_leaf,
            max_features: w.max_features.into(),
            min_impurity_decrease: w.min_impurity_decrease,
        }
    }
}

impl From<&RandomForestParams> for ParamsWire {
    fn from(p: &RandomForestParams) -> Self {
        ParamsWire {
            tree: (&p.tree).into(),
            n_estimators: p.n_estimators,
            bootstrap: p.bootstrap,
            max_samples: p.max_samples.into(),
        }
    }
}
impl From<ParamsWire> for RandomForestParams {
    fn from(w: ParamsWire) -> Self {
        RandomForestParams {
            tree: w.tree.into(),
            n_estimators: w.n_estimators,
            bootstrap: w.bootstrap,
            max_samples: w.max_samples.into(),
        }
    }
}
