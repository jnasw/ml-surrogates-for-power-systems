# PINN vRBA Stage 0/1 Design Note

This note records the Stage 0/1 decisions for integrating a paper-aligned vRBA
mechanism into the PINN training stack without changing current behavior.

## Scope Frozen For The First Functional Milestone

- Primary target set: `physics`
- Future extension targets: `ic`, `data`, `dt`
- Mechanisms to support eventually:
  - adaptive collocation sampling
  - adaptive local residual weighting
  - a shared state that can drive both
- First functional potential to prioritize: `quadratic`
- Second functional potential to add later: `exponential`

## Representation Choice

The future vRBA implementation should maintain persistent state over a stable
discrete point set. In this repo, that should mean a persistent collocation
pool, not weights tied only to transient minibatches.

This is the closest practical match to the paper and the `other_repos/vRBA`
PINN mechanism.

## Phase Interaction Policy

The current default design assumption is:

- stochastic / minibatch phases may update vRBA state
- full-batch / closure-based phases should be allowed to freeze sampling and
  weighting by default

Stage 0/1 only records this decision in configuration. No training behavior is
altered yet.

## Stage 0/1 Deliverables

- dedicated `pinn.vrba` config block
- serializable vRBA config and state containers
- dormant checkpoint payload support
- dormant metrics/logging fields

No collocation strategy, loss computation, or optimizer behavior is changed in
Stage 0/1.
