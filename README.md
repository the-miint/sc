# sc

A pure-Rust reimplementation of the RandomForest classification and regression
workflows from [q2-sample-classifier](https://github.com/qiime2/q2-sample-classifier),
exposed as a reentrant C-API centered on Apache Arrow for C/C++ consumers.

- **Scope (v1):** RandomForest classifier + regressor; fit/predict, class
  probabilities, feature importances, RFECV feature selection, randomized
  hyperparameter tuning, nested/out-of-fold cross-validation, and (v1/v1.1)
  TreeSHAP explanations.
- **Consistency:** statistically equivalent to q2-sample-classifier (not
  byte-exact), validated against committed oracle fixtures generated from q2.
- **Pure Rust**, no C/C++/Fortran dependencies; WASM-friendly.

Design and milestone plan live alongside this project (see `sc-design.html` and
`sc-implementation-plan.html`). Implementation proceeds milestone by milestone
using red/green/refactor TDD with a code review and written summary at each
milestone boundary.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
