# Blind-first human oversight protocol

Human review is not independent when the reviewer sees the AI recommendation
before forming an initial judgment. Measuring only the final answer cannot tell
whether agreement reflects genuine corroboration or anchoring.

`app.oversight_protocol` defines a narrow, fail-closed trace contract for a
blind-first workflow:

1. the service assigns a versioned case;
2. the reviewer commits an independent decision and confidence;
3. the interface reveals the content-addressed AI recommendation;
4. the reviewer commits the final decision and records an approved reason if it
   differs from the independent judgment.

Each event is ordered, timestamped and chained to the preceding event digest.
The audit independently enforces chronology, future skew, stage deadlines and
the total session lifetime. Unknown fields, duplicate JSON keys, non-finite
values, unsupported decisions and resource-budget violations fail closed.

## Evidence and privacy

The report exposes stable reason codes, stage latencies, a policy digest and the
digest of the complete trace. Session, case and reviewer identifiers are
one-way hashed before they enter the report; raw decision narratives and AI
content are represented only by producer-supplied SHA-256 identities.

These hashes are integrity identifiers, not encryption. Low-entropy identifiers
remain guessable, so the report requires the same access controls as other
review telemetry.

## CLI

```bash
python -m app.oversight_protocol trace.json \
  --now 2026-09-29T00:00:00+00:00 \
  --output oversight-report.json
```

| Exit | Meaning |
|---:|---|
| `0` | trace satisfies the protocol |
| `2` | well-formed trace violates protocol policy |
| `3` | malformed, over-budget or unreadable artifact |

`--now` makes historical replay deterministic. Production callers should omit
it and use the service clock.

## Enforcement boundary

The reveal endpoint should be a server-side state transition: it must require a
successfully persisted independent-decision event, append the reveal event in
the same durable session, and return the recommendation only after commit. A UI
flag or client-provided “blind” boolean is not evidence of independence.

The digest chain detects changes relative to the supplied trace, but it does not
authenticate the producer or prove the UI withheld the recommendation. A
database attacker can recompute an unsigned chain. Production deployment should
persist events append-only, authenticate reviewer identity, sign or MAC terminal
checkpoints, and bind model/prompt revisions to an immutable registry.

The next implementation step is a transactional reveal endpoint backed by the
existing experiment API, followed by fault-injection tests for concurrent reveal
and independent-decision writes.
