# 🧠 Prompt Library & Techniques

> Reusable prompts + the prompt-engineering techniques used to build this project,
> documented Anthropic-style. Use these to keep driving the project the same way.
> Each technique answers the **W-forms** (What · Why · Who · When · Where · Which ·
> How). Prompts follow a single template (below).

---

## The template (Anthropic-style)
Wrap intent in XML-ish tags; be clear and direct; give context; show the output
shape; add examples when format matters. (Mirrors Anthropic's "clear & direct +
role + context + examples + let-it-think + XML tags" guidance.)

```
<role>Who the model should act as.</role>
<context>What it needs to know (repo paths, constraints, prior decisions).</context>
<task>The single, concrete objective.</task>
<constraints>Hard rules (offline-verifiable, lazy imports, don't delete docs…).</constraints>
<output_format>Exactly what to return (files, a table, a diff, a plan).</output_format>
<examples>Optional: a worked example of the expected output.</examples>
```

---

## Techniques used (with W-forms)

### 1. Role prompting
- **What:** assign an expert persona ("act as a solution architect / staff engineer").
- **Why:** steers tone, depth, and judgement toward expert-level trade-offs.
- **Who/When/Where:** model; at design/decision moments; in `<role>`.
- **Which/How:** one sentence at the top: *"You are a senior ML platform architect…"*.

### 2. Clear & direct + constraints
- **What:** state the one objective + explicit hard rules.
- **Why:** removes ambiguity; prevents scope creep and wrong assumptions.
- **When/Where:** every task; `<task>` + `<constraints>`.
- **How:** imperative voice; enumerate non-negotiables ("must run offline", "lazy-import torch").

### 3. Decomposition / prompt chaining
- **What:** split a big goal into phases, each its own prompt (Phase 0→6 here).
- **Why:** smaller verifiable units; the model stays accurate; you stay in the loop.
- **When:** any multi-step build.
- **How:** plan phases first, then run one phase per prompt; feed results forward.

### 4. Explore → Plan → Code → Commit (agentic loop)
- **What:** read/locate first, design, implement, verify, ship — in that order.
- **Why:** grounds changes in the real codebase; avoids blind edits.
- **Who:** sub-agents (Explore/Plan) + main loop.
- **How:** "explore X and report a reuse map" → "design Y" → "build + test" → "commit".

### 5. Multi-agent fan-out
- **What:** launch parallel Explore/Plan agents for independent areas.
- **Why:** faster, broader coverage; isolates context.
- **When:** scope is uncertain or spans many files.
- **How:** spawn ≤3 agents with distinct, specific briefs; synthesize results.

### 6. Few-shot / templates
- **What:** provide a template/example to fix the output shape (RFC/PRD/ADR docs).
- **Why:** consistent, reviewable artifacts; less rework.
- **Where:** `docs/specs/TEMPLATE_*.md`, this file.
- **How:** "fill this template" beats "write a doc".

### 7. Let-it-think (chain of thought)
- **What:** ask for reasoning/trade-offs before the answer.
- **Why:** better decisions; surfaces alternatives (every RFC has an Alternatives section).
- **How:** "think step by step", "list alternatives and why rejected".

### 8. Verification loops (test-before-claim)
- **What:** run tests / build after every change; never claim done unverified.
- **Why:** truthful status; catches regressions immediately (42 pytest + Next build).
- **How:** "implement, then run pytest and paste results; only then commit".

### 9. Reuse-first constraint ("don't reinvent")
- **What:** prefer OSS / existing code over custom.
- **Why:** less to maintain, higher quality (supervision, Roboflow, TrackNet).
- **How:** "search for an existing lib/model; only write custom for glue".

### 10. Structured output / schemas
- **What:** force a machine-checkable shape (JSON schema, file:line table).
- **Why:** reliable downstream use; no parsing guesswork.
- **How:** give the exact schema; validate.

---

## Reusable prompts

### P1 — Ship a feature (use the `ship-feature` skill)
```
<role>Senior engineer on the Pickleball-Vision-LLM managed-stack SaaS.</role>
<context>FastAPI app/ + Modal worker/ + Supabase + Next.js web/. Patterns: lazy
heavy-imports, injectable services with Fake impls, src/ is the reusable core.</context>
<task>Add <FEATURE>.</task>
<constraints>Offline-verifiable (pytest + npm build); lazy-import any heavy/cloud
dep; mirror the closest existing module; update docs/TASKS.md; one commit + push.</constraints>
<output_format>The code files, the test, the passing test output, and the commit.</output_format>
```

### P2 — Plan a phase / research → RFC
```
<role>Solution architect.</role>
<context><AREA> in this repo. Read the relevant files first.</context>
<task>Research options (web ok) and produce a design.</task>
<constraints>Use the RFC template (docs/specs/TEMPLATE_RFC.md); include Alternatives
+ Risks + References with links; right-size for ~200 customers (managed-first).</constraints>
<output_format>docs/specs/RFC-NNN-*.md filled, + an ADR if a real decision is made.</output_format>
```

### P3 — Verify / review
```
<role>Skeptical reviewer.</role>
<context>The diff/branch for <CHANGE>.</context>
<task>Find correctness bugs + reliability/security gaps.</task>
<constraints>Run the tests; verify claims; one line per finding (path:line, severity, fix).</constraints>
<output_format>Severity-tagged findings; then a verdict (ship / fix-first).</output_format>
```

### P4 — Infra / cost analysis
```
<role>DevOps/platform architect.</role>
<context>Managed stack (Modal/Supabase/Vercel/Stripe/Bedrock), ~200 customers.</context>
<task>Give min→max infra with DevOps effort, compute, and cost per tier.</task>
<constraints>System-arch view; don't over-build for the scale; cite pricing pages.</constraints>
<output_format>A tier table + a "how to choose" note (see docs/INFRA_SCALING.md).</output_format>
```

## How to use
Pick the prompt, fill `<...>`, paste. For features, just say *"use the ship-feature
skill to add <X>"* — it encodes P1 + the project patterns. See also
`.claude/docs/DECISIONS_LOG.md` for why the project is shaped this way.

## References
- Anthropic prompt engineering: clear-&-direct, multishot, chain-of-thought, XML
  tags, role/system prompt, prefill, prompt chaining
  ([platform.claude.com/docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview)).
- Internal: `.claude/skills/ship-feature/SKILL.md` · `docs/specs/TEMPLATE_RFC.md` ·
  `docs/specs/RESEARCH_NOTES.md`.
