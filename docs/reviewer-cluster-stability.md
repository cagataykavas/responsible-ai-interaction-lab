# Reviewer-cluster stability audit

Repeated decisions from one reviewer are correlated. Treating them as independent observations can
make a human-agent interface effect look far more certain than its reviewer coverage supports.
`app.reviewer_cluster_audit` adds a fail-closed release boundary for paired binary evaluation
outcomes such as correct decisions, harmful agreements, or successful interventions.

The audit:

- validates unique decisions and bounded reviewer clusters;
- rejects evidence with too few observations or reviewers;
- limits the fraction of evidence contributed by one reviewer;
- resamples complete reviewers in a deterministic cluster bootstrap;
- gates confidence-interval width;
- measures every leave-one-reviewer-out effect and gates the largest shift; and
- requires effect-direction stability when the observed effect exceeds a practical floor.

Reports contain aggregate rates and hashed reviewer references for the ten most influential
clusters. A SHA-256 digest binds the exact decisions and policy without copying raw reviewer IDs
into the report.

Input is a strict JSON array:

```json
[
  {
    "decision_id": "decision-001",
    "reviewer_id": "reviewer-17",
    "baseline_success": false,
    "treatment_success": true
  }
]
```

```bash
python -m app.reviewer_cluster_audit paired-decisions.json \
  --min-reviewers 8 \
  --max-reviewer-share 0.25 \
  --max-leave-one-reviewer-shift 0.15 \
  --output reviewer-cluster-report.json
```

Exit code `0` means the evidence passed, `2` means valid evidence failed policy, and `3` means the
evidence or operation was invalid. Parsing rejects duplicate keys, unknown fields, non-finite
values, symlinks, oversized inputs, and excessive bootstrap work. Output uses an fsynced temporary
file and atomic replacement.

## Limitations

The cluster bootstrap assumes reviewers are independent clusters and that paired outcomes are
comparable within the declared experiment. It does not correct randomization failures, missing
outcomes, carryover, topic or time dependence, reviewer learning, or unmeasured confounding. The
binary success metric must be defined before evaluation. The next step is a pre-registered,
multiway cluster analysis over reviewer, task family, and time block with authenticated assignment
and outcome manifests.
