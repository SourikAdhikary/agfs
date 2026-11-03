## Summary

- **Subsystem / Layer (`core` | `storage` | `engine` | `memory` | `safety` | `adapters` | `interfaces`):**
- **Research Paper / Invariant Referenced (if applicable):**

## Verification Checklist

- [ ] `make check` (`ruff check src benchmarks tests`) passes with 0 warnings
- [ ] `make test` passes (unit, integration, chaos, and layer-boundary tests)
- [ ] `make doctor` passes all **12 formal kernel invariants (`INV-01`..`INV-12`)**
- [ ] `make bench` confirms the change measurably improves or preserves A/B benchmark metrics (`Without AgFS` vs. `With AgFS`)
