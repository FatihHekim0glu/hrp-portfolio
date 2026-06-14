# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-14

### Added

- Initial package skeleton (src-layout, import name `hrp`).
- Core helpers: `_constants`, `_typing`, `_exceptions`, `_validation`,
  `_manifest` (`RunManifest` with BLAKE2b config-hash), and `_rng`
  (seeded PCG64 generator + substream spawning).
- Stub signatures with full contracts for the estimator (`covariance`, `rmt`,
  `mu`), cluster (`distance`, `linkage`, `quasidiag`), allocation
  (`hrp`, `ivp`, `naive`, `markowitz_adapter`), backtest
  (`walk_forward`, `costs`, `stats`), and evaluation
  (`dsr`, `comparison`, `verdict`) subpackages.
- Plotly figure builders, data loaders, and Typer CLI stubs.
- Seeded synthetic test fixtures (one-factor, block-correlation, pure-noise,
  singular-covariance, de Prado worked example).

[Unreleased]: https://github.com/FatihHekim0glu/hrp-portfolio/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/hrp-portfolio/releases/tag/v0.1.0
