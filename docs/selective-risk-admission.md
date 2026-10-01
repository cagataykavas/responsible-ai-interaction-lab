# Selective-risk admission

A high average model accuracy does not establish that an automated slice is safe. The model may
retain most of its errors among the cases it labels “high confidence,” while deferring easy cases.
This evaluator separates three operational questions:

- **coverage** — what fraction of cases will be automated at a fixed confidence threshold;
- **selective risk** — the error rate inside that automated fraction;
- **deferred error capture** — what fraction of all model errors are routed away from automation.

It also computes the area under the risk-coverage curve (AURC). The curve admits complete
equal-confidence groups, preventing arbitrary case order from improving the result.

```python
from app.selective_risk import SelectiveRiskPolicy, audit_selective_risk

policy = SelectiveRiskPolicy(
    confidence_threshold=0.80,
    minimum_coverage=0.45,
    maximum_selective_risk=0.10,
    maximum_aurc=0.20,
)
report = audit_selective_risk(holdout_predictions, policy)
if not report.accepted:
    raise RuntimeError(report.violations)
```

The policy must be registered before evaluating the holdout set. Selecting the threshold or limits
on the same observations and then reporting the result as a release test would be optimistic. For
time-dependent data, the holdout must also follow training and calibration chronologically.

Run the deterministic synthetic example:

```bash
python -m app.selective_risk
```

## Interpretation limits

This gate measures model errors under a confidence-based automation rule. It does not estimate how
deferred cases will perform with human review, prove probability calibration, establish subgroup
fairness, or account for unequal error costs. A production policy should pair it with calibrated
probabilities, cost-sensitive and subgroup checks, temporal validation, drift monitoring, and a
measured human-review fallback. SHA-256 evidence binds the evaluated rows but does not authenticate
the producer.
