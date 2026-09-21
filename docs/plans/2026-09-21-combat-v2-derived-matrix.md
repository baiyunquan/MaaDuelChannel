# Combat v2 Derived Matrix Implementation Plan

## Goal

Replace the heuristic combat feature path with one versioned feature builder shared by
training and inference. The builder emits raw unit features, twelve formation features,
twenty directed relation features, and explicit validity metadata.

## Task 1: Combat knowledge contract and parser

- Add structured opening attacks, controls, healing actions, future-form summaries,
  provenance, revision, and confirmed/estimated/missing status.
- Upgrade the contract to `combat-v2` and keep legacy JSON readable for migration.
- Parse map overrides before generic enemy-page data and avoid keyword-only capability
  classification.
- Add parser and damage-action contract tests before implementation.

## Task 2: Battlefield calibration and battle-state contract

- Add a homography calibration schema with fit/check points, hashes, error validation,
  raw image coordinates, tile coordinates, quality, and calibration version.
- Introduce an unlabeled `BattleState`; keep `winner` only on training samples.
- Require a compatible calibration when raw video coordinates must be transformed.
- Add identity/perspective/error-threshold tests before implementation.

## Task 3: Derived combat and formation features

- Implement FP32 physical, arts, and true damage formulas per hit.
- Emit the fixed 20 relation channels plus validity, estimate, targetability, and finite
  masks.
- Implement target-capacity allocation, opening damage, control uptime, healing,
  counterfactual debuff gain, screening, protection time, and the fixed 12 formation
  summaries.
- Add monotonicity, multihit, capacity, permutation, grouping, and screening tests before
  implementation.

## Task 4: Relation Transformer and training integration

- Feed unit, formation, position, side, and ID encodings into the transformer.
- Consume the precomputed directed relation matrix in relation attention.
- Preserve antisymmetric left/right scoring and padding/permutation invariance.
- Bind checkpoints and dataset contracts to knowledge, formula, calibration, feature,
  and vocabulary versions and hashes.
- Add one small forward/backward and checkpoint compatibility test before implementation.

## Task 5: Inference and explainability

- Add `predict-duel` for a label-free battle-state JSON input.
- Reuse the same feature builder and reject contract mismatches.
- Add `--explain` output for DPS, TTK, coverage, target allocation, and protection
  relationships with instance IDs.
- Add a checkpoint-load inference test and CLI help test before implementation.

## Verification

Run targeted tests after each task, then run the full unit suite and Ruff. Do not start
the formal full-data GPU training in this change; only run a small forward/backward and
checkpoint inference check.
