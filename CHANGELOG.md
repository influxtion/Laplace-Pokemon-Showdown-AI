# Changelog

## 1.1.0 — 2026-11-07

### Added

- One-command setup from a checkout, including the Terastallization-enabled Rust
  engine, CPU PyTorch, Random Battle set data, and offline validation.
- Linux and Windows CI, engine/model smoke tests, replay-analysis regression
  checks, and a redacted Git-history secret scan.
- An animated decision-trace demo, links to public battle replays, and a social
  preview image asset.
- Contributor instructions, issue forms, a pull request template, and an example
  credentials file.

### Changed

- `main` contains the standalone Random Battle bot; the full OU development
  version is preserved on `OU`.
- Project license changed from Apache-2.0 to MIT for this release onward.
- README now leads with what the bot does, a demo, and the shortest setup path.

This release improves setup, documentation, and contributor support. It does not
claim a new ladder result or change the random-team search policy.

## 1.0.0 — Previously untagged

- Gen 9 Random Battle bot with determinized MCTS, a trained value head, mixed
  action selection, replay archiving, and loss analysis.
- Reported historical peak: 2231 Elo / approximately 83.3 GXE.
