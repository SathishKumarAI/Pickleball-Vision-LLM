# 📐 Specs — PRD & Design Docs

Structured product + engineering specs for Pickleball-Vision-LLM, using the
recognized **PRD** (product) and **RFC / design-doc** (engineering) templates that
high-output software teams use. Not a claim to any private internal Anthropic
template — this is the widely-adopted public structure (Tanya Reilly / Pragmatic
Engineer for RFCs; standard PRD sections), which is the shape Anthropic-style teams
work in.

## Contents
| Doc | Purpose |
|---|---|
| `TEMPLATE_PRD.md` | Blank reusable PRD template |
| `TEMPLATE_RFC.md` | Blank reusable design-doc / RFC template |
| `PRD-pickleball-vision.md` | Filled PRD — the product |
| `RFC-001-video-analysis-pipeline.md` | Filled design doc — pipeline & product backend |
| `RFC-002-aws-infrastructure.md` | AWS infra & IaC (Terraform) — **superseded by RFC-003** |
| `RFC-003-managed-stack.md` | **Current** — managed stack (Modal + Supabase + Next.js + Stripe) |
| `RESEARCH_NOTES.md` | Cited research dossier (CV models, tracking, action recognition, court homography, AWS, doc templates) |

## How to use
1. New feature/product → copy `TEMPLATE_PRD.md`, fill it, circulate for review.
2. New system/change → copy `TEMPLATE_RFC.md` as `RFC-NNN-title.md`, fill, name
   3–5 required reviewers, set a 2-week review deadline.
3. Keep docs ≤5–7 pages; link out for deep detail; use diagrams.

## Conventions
- RFCs numbered `RFC-NNN-kebab-title.md`, status in the header
  (`Draft → In Review → Accepted → Implemented → Superseded`).
- Every RFC must include **Alternatives Considered** — omitting it signals the
  problem wasn't thought through.
- **Goals** define success; **Non-Goals** prevent scope creep — both required.

Sources: [Pragmatic Engineer — RFCs & Design Docs](https://blog.pragmaticengineer.com/rfcs-and-design-docs/) ·
[Product School — PRD template](https://productschool.com/blog/product-strategy/product-template-requirements-document-prd)
