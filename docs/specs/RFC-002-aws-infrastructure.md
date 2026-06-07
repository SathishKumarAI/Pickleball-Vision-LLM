# RFC-002: AWS Infrastructure & IaC

| | |
|---|---|
| **Status** | Draft |
| **Author** | Sathish Kumar |
| **Reviewers** | TBD (eng lead, infra/DevOps, ML lead) |
| **Created / Updated** | 2026-06-07 |
| **Review deadline** | TBD |
| **Depends on** | RFC-001 (pipeline design), `docs/DELIVERY_PLAN.md`, `docs/BUDGET_PLAN.md` |

> ⚠️ **SUPERSEDED by [RFC-003](RFC-003-managed-stack.md)** (2026-06-07). At the
> target scale (~200 customers) this self-managed AWS design is over-built; the
> product now uses a managed stack (Modal + Supabase + Next.js + Stripe + Bedrock).
> Kept for reference / future high-scale (>10k/mo) consideration. See [ADR-0005](../adr/0005-managed-stack.md).

## Summary
Cloud infrastructure for the video-analysis product on **AWS**, provisioned as
code. Stateless Flask API on Fargate behind an ALB; durable state in Redis +
Postgres; a Celery task queue; a **GPU worker pool on ECS-on-EC2 g5 Spot** that
autoscales on queue depth; S3 + CloudFront for media. **IaC: Terraform** (with CDK
considered). Sized for ~10k videos/mo.

## Context & problem statement
RFC-001 defines the app + pipeline; today it runs single-box with in-process state.
To be durable, scalable, and observable we need real AWS infra — but provisioned
reproducibly (no click-ops), with staging↔prod parity, spot-cost efficiency, and a
GPU pool that scales with load without OOM. The hard parts: GPU autoscaling (GPU
util isn't a built-in scaling metric), Spot interruption handling, and IaC tool
choice.

## Goals
- One-command, reproducible provisioning of all envs (staging, prod).
- API autoscales on load; **GPU workers autoscale on queue depth**, Spot-first with
  On-Demand fallback.
- Durable jobs (Redis) + users (Postgres); media in S3 with signed-URL delivery.
- Least-privilege IAM, secrets out of code, private data subnets.
- Observability: latency p95, GPU util, queue depth, error rate, alarms.
- Cost ≈ budget ($270/mo @10k; rule-only ≈ $120) — spot GPU + lifecycle TTL.

## Non-Goals
- Multi-region/DR (single region first). · Kubernetes/EKS (ECS is enough at this
  scale). · Multi-cloud deploy (Terraform keeps the door open; not built now).
- App/pipeline logic (RFC-001).

## Proposed design

### Topology
```
                     Route53 ─ ACM(TLS)
                          │
Internet ─▶ CloudFront ─▶ ALB ─▶ ECS Fargate: API (Flask, 2+ tasks, public subnets)
                                   │            │
                                   │            ├─▶ ElastiCache Redis  (jobs + Celery broker)   [private]
                                   │            ├─▶ RDS Postgres       (users)                  [private]
                                   │            └─▶ S3 (uploads/outputs) + presigned URLs
                                   │
                          Celery `gpu` queue (Redis)
                                   │
                ECS-on-EC2 ASG: GPU workers (g5.xlarge, Spot-first)  [private]
                  • Capacity Provider: Spot priority, On-Demand fallback
                  • scale-out on queue depth (CloudWatch), scale-in on idle
                  • DCGM-exporter → GPU util metric; models pre-warmed (AMI/EBS cache)
                                   │
                            S3 outputs ◀── annotated.mp4 + result.json
                                   │
                          Bedrock (Claude Haiku) for cloud LLM (post-MVP)
```

### Networking
- **VPC** with public subnets (ALB, NAT) + private subnets (API tasks, workers,
  RDS, Redis). Workers/data have no public IPs; egress via NAT (or VPC endpoints
  for S3/Bedrock/ECR to cut NAT cost).
- Security groups: ALB→API:80/443; API→Redis:6379, RDS:5432; workers→Redis, S3
  (endpoint), Bedrock.

### Compute
- **API:** ECS **Fargate** service, target-tracking autoscale on ALB
  request-count/CPU. 2+ tasks across AZs.
- **GPU workers:** ECS-on-**EC2 g5.xlarge** in an **Auto Scaling Group** behind an
  **ECS Capacity Provider** (Spot priority, On-Demand fallback for uptime). Worker
  Celery concurrency = real GPU slots (start 1). **Pre-warm models** baked into the
  AMI or cached on an EBS volume to keep cold-load out of the latency budget.
- **Scaling signal:** queue depth (Celery/Redis length or SQS
  `ApproximateNumberOfMessages`) via CloudWatch → target-tracking on the ASG. GPU
  util (DCGM-exporter) as a secondary/observability metric (not a built-in scaling
  metric).

### Data & storage
- **RDS Postgres** (small, Multi-AZ in prod) — users; replaces SQLite `UserDB`.
- **ElastiCache Redis** — job state (TTL keys) + Celery broker/result backend.
- **S3** — `uploads/`, `jobs/<id>/annotated.mp4`, `result.json`; **lifecycle TTL**
  for retention/GDPR; **presigned URLs** + CloudFront for download.
- Model weights in S3, versioned keys, cached on worker volume.

### Security & secrets
- **IAM** least-privilege task roles (API: Redis/RDS/S3; worker: S3/Bedrock).
- **Secrets Manager / SSM Parameter Store** for `APP_SECRET`, DB creds, Bedrock
  config — injected as env/secret refs, never in image.
- TLS via ACM at ALB/CloudFront. WAF on CloudFront (rate-limit, basic rules).

### Observability
- CloudWatch metrics/logs: API p95 latency, per-stage pipeline timings (emitted by
  worker), GPU util (DCGM), queue depth, jobs-by-status. Alarms near SLA + on Spot
  interruption rate. (Optional Prometheus/Grafana — configs already in `deployment/`.)

### Environments & CI/CD
- `staging` + `prod` from the same Terraform modules (workspace/var per env).
- CI builds + pushes images to **ECR**; deploy via `terraform apply` (plan-gated)
  and ECS rolling update. DB migrations as a one-off ECS task.

### IaC decision — **Terraform** (recommended)
- HCL, declarative, mature AWS provider; **remote state in S3 + DynamoDB lock**.
- Module layout: `network/`, `data/` (rds, redis, s3), `api/` (ecs-fargate, alb),
  `workers/` (asg, capacity-provider, launch-template), `observability/`, `iam/`.
- Cloud-agnostic hedge (we already consider Azure/Vertex for LLM).

## Alternatives considered
- **AWS CDK** — TypeScript/Python, auto state via CloudFormation, deep AWS
  integration, dev-friendly for an AWS-centric team. Rejected as default for the
  multi-cloud hedge + Terraform's larger ecosystem/state control; **revisit if the
  team prefers code-over-HCL** (both can coexist — hybrid IaC is valid).
- **CloudFormation (raw)** — verbose, AWS-only; CDK supersedes it.
- **EKS/Kubernetes** — overkill at 10k/mo; ECS is simpler. Revisit at large scale.
- **Fargate for GPU** — Fargate has no GPU; GPU must be ECS-on-EC2 (or SageMaker).
- **SageMaker async inference** — viable for the model step, but couples us to its
  packaging; ECS workers keep the existing `Pipeline` code path. Reconsider for
  managed scaling later.
- **SQS vs Celery/Redis** — SQS is fully managed + native CloudWatch depth metric;
  Celery/Redis reuses the app code + richer task semantics. Start Celery/Redis;
  SQS is a drop-in if we want managed.
- **On-Demand only GPU** — simpler, ~3× cost. Rejected; Spot + fallback covers it.

## Trade-offs & risks
- **Spot interruptions** — jobs must be idempotent + requeue on interruption
  (`acks_late`, content-hash cache from RFC-001/P1-3); 2-min handler drains in-flight.
- **GPU autoscaling lag** — cold scale-out + model load can miss the budget under
  bursts → keep a warm minimum (1 worker), pre-warm AMI, queue absorbs spikes.
- **State management (Terraform)** — needs S3+DynamoDB backend + discipline; drift if
  someone click-ops. Enforce plan-gated applies.
- **NAT/data-transfer cost** — mitigate with S3/Bedrock/ECR VPC endpoints.
- **RDS/Redis as SPOFs** — Multi-AZ in prod.

## Security / privacy / cost
- Private subnets for data+workers; least-privilege IAM; secrets managed; WAF+TLS.
- Faces → S3 lifecycle TTL + `DELETE /jobs/<id>` (RFC-001 P1-5); in-region only.
- Cost levers: g5 **Spot** (70–90% off), warm-min=1, lifecycle TTL, VPC endpoints,
  Haiku + rule fallback. Track vs `docs/BUDGET_PLAN.md`.

## Rollout & testing plan
- Maps to `docs/DELIVERY_PLAN.md` sprints: S2 (network+RDS+Redis+S3+API Fargate),
  S3 (Celery+GPU ASG+capacity provider+autoscaling), S4 (observability+lifecycle).
- `terraform plan` in CI on PRs; apply to **staging** first, smoke `/health` +
  one real job, then **prod**. Load-test queue-depth scaling. Rollback = previous
  task-def + `terraform apply` of last good state.

## Open questions
- Terraform vs CDK final call (team preference). · SQS vs Celery/Redis for the queue.
- Warm-minimum worker count (cost vs burst latency). · Single vs Multi-AZ data in
  staging. · Retention window (S3 TTL days).

## References / Further reading
- IaC: [Spacelift — CDK vs Terraform](https://spacelift.io/blog/aws-cdk-vs-terraform) · [Pluralsight — CloudFormation/Terraform/CDK](https://www.pluralsight.com/resources/blog/cloud/cloudformation-terraform-or-cdk-guide-to-iac-on-aws) · [Terraform S3 backend (state+lock)](https://developer.hashicorp.com/terraform/language/settings/backends/s3)
- ECS GPU autoscaling: [aws-samples/ecs-gpu-scaling](https://github.com/aws-samples/ecs-gpu-scaling) · [Deploying GPU ECS EC2 with ASG + launch templates](https://dev.to/bikash119/deploying-gpu-enabled-ecs-ec2-instances-with-auto-scaling-groups-and-launch-templates-569l) · [EC2 GPU best practices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configure-gpu-instances.html)
- Spot/capacity: [ECS Capacity Providers](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/cluster-capacity-providers.html) · [EC2 Spot best practices](https://www.cloudexmachina.io/blog/aws-spot-instances) · [Scale GPU Spot on S3 upload](https://matt.sh/aws-ec2-autoscale-gpu-spot-s3-uploads)
- Services: [ECS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/userguide/what-is-fargate.html) · [ALB](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html) · [RDS Postgres](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_PostgreSQL.html) · [ElastiCache Redis](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/WhatIs.html) · [S3 presigned URLs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html) · [CloudFront](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Introduction.html)
- LLM: [Invoke Claude on Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
- Internal: `docs/specs/RFC-001-video-analysis-pipeline.md` · `docs/DELIVERY_PLAN.md` · `docs/BUDGET_PLAN.md` · `docs/REMEDIATION_PLAN.md`
