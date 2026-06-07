# PRD: Pickleball-Vision-LLM — AI Game Analysis

| | |
|---|---|
| **Status** | Draft |
| **Author** | Sathish Kumar |
| **Reviewers** | TBD (eng lead, ML lead, product) |
| **Created / Updated** | 2026-06-07 |
| **Review deadline** | TBD |

## 1. Problem / Opportunity
Pickleball is the fastest-growing US sport, but recreational and competitive
players have **no affordable, automated way to analyze their game**. Coaching is
expensive and not scalable; existing tools (PB Vision, SwingVision) are paid/closed.
A player who records a match on a phone has no easy path from raw footage to
actionable feedback. **Opportunity:** upload a video, get back an annotated video +
coaching insights in minutes — democratized sports analytics.

## 2. Target users / personas
- **Recreational player** — wants simple feedback ("where am I losing points?").
- **Competitive player / coach** — wants positioning, shot, and rally analytics.
- **Club / league** — batch analysis, highlights, player profiles (later).

## 3. Goals
- Turn an uploaded match clip into an **annotated output video** (players, ball,
  track IDs, action labels) + a **coaching report**.
- Deliver results **within minutes** of upload (≤2-min clip → result in ~≤2 min).
- Self-serve web product behind login; users own their uploads and results.
- Run on commodity GPU + managed cloud; cost-efficient at ~10k videos/mo.

## 4. Non-Goals
- Real-time / live-broadcast analysis (this phase is post-upload).
- Mobile/wearable apps; proprietary broadcast feeds.
- Auto-officiating / line-calling as a product guarantee.
- Multi-sport (pickleball first; architecture allows extension later).

## 5. Success metrics (KPIs)
- **Latency:** p95 end-to-end ≤ 2× clip duration; ≥95% of 2-min clips < ~120 s.
- **Reliability:** ≥99% jobs reach `done` or a clear `error` (no silent loss).
- **Quality:** detection mAP ≥ baseline Roboflow weights; coaching rated "useful"
  ≥70% (rubric / thumbs-up).
- **Cost:** ≤ ~$0.05 / video blended (see `docs/BUDGET_PLAN.md`).
- **Activation:** % of registered users who complete ≥1 analysis.

## 6. Requirements / Features (MoSCoW)
- **Must:** login/auth · video upload + validation · async job + progress · annotated
  output video · rule-based coaching report · download results · per-user ownership.
- **Should:** court-aware positioning (homography) · action labels · job history ·
  cloud LLM coaching (Bedrock) · heatmaps.
- **Could:** rally segmentation/highlights · player profiles over time · webhooks.
- **Won't (now):** live analysis · mobile app · multi-sport · social/sharing.

## 7. Non-functional requirements
- **Performance:** ≤2-min latency budget (FRAME_SKIP, 720p cap, GPU batch).
- **Scale:** ~10k videos/mo, autoscaling GPU pool on AWS.
- **Security:** auth + ownership; rate-limited auth; signed download URLs.
- **Privacy/compliance:** videos contain faces → retention policy + delete endpoint
  (GDPR); data stays in-region.
- **Availability:** stateless API; durable job state; crash → requeue.

## 8. User flows
1. Register / log in.
2. Upload a match clip → immediate `job_id` (validated: size/duration/codec).
3. Poll progress (queued → running %→ done).
4. Download annotated video + coaching report; view in dashboard.
5. (Later) browse past analyses; delete data.

## 9. Open questions
- Retention window for uploads (sets S3 TTL). · Coaching depth for v1 (rule vs cloud
  LLM — decided: rule-first). · Free vs paid tiers / quota.

## 10. Milestones / rollout
Per `docs/ROADMAP.md` (M0 seam → M5 UX) and `docs/DELIVERY_PLAN.md` (6 AWS sprints).
Rule-coaching MVP first; cloud LLM (Bedrock Haiku) post-MVP.

## References / Further reading
- [Product School — PRD template & sections](https://productschool.com/blog/product-strategy/product-template-requirements-document-prd)
- [Perforce — How to write a PRD](https://www.perforce.com/blog/alm/how-write-product-requirements-document-prd)
- [Atlassian/Confluence — PRD template](https://www.atlassian.com/software/confluence/templates/product-requirements)
- Internal: `docs/ROADMAP.md`, `docs/DELIVERY_PLAN.md`, `docs/BUDGET_PLAN.md`, `docs/MODELS_AND_REUSE.md`
- Market refs: [PB Vision](https://pb.vision/), [SwingVision](https://swing.vision/)
