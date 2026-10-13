# Use of AI assistants (guide §23.5)

Neuro3D-Lab is a project of Francisco Angulo de Lafuente, who is responsible for its content. AI systems are tools here, not authors. Major publishers do not accept AI systems as authors and ask for their use to be disclosed; any manuscript from this repository must follow the policy of the journal it is sent to, checked at submission time.

## Which assistants did what

| Assistant | Role | Where it is recorded |
|---|---|---|
| Claude (Anthropic; Claude Code) | Wrote most of the code, the contracts, the analyses and the documentation under the owner's instructions; ran the experiments | Git history (commits carry a `Co-Authored-By: Claude` trailer), `coordinacion/` |
| Codex (OpenAI) | Independent audit of the core and the add-on (task T-001), corrections (T-002) | `coordinacion/respuestas/` |
| A separate Claude agent with no part in the drafting | Second-opinion review of the CONTRATO-013/014 pre-registrations before freezing; all points were addressed before registration | CONTRATO-013, status paragraph |
| JEV (the owner's decision model, reached through a local router) | Advisory answers on protocol and compute decisions, recorded with provenance and confidence | `evidence/jev-protocol-*.json`, contract texts |
| The completion guide (written with AI assistance for the owner) | Protocols, criteria, reference code; its numbers were re-executed here, not trusted | `docs/CONTRACTS/CONTRATO-012-numerical-closure.md` |

## Rules applied

- An answer from an assistant is never a substitute for a proof or a measurement. Every quantitative statement points to an executed artifact.
- The agent that designs a protocol does not judge its own results alone: pre-registrations were reviewed by a separate agent; external human review and replication are still pending and are not claimed.
- Contract thresholds were fixed before the corresponding runs. Changes of plan are recorded as amendments with reasons, never by editing earlier records.
- No credentials, personal data or private keys were given to any assistant.
