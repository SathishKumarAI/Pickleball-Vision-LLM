"""FastAPI application package for Pickleball-Vision-LLM.

The production control plane (CPU, stateless): verifies Supabase JWTs, manages
jobs/quotas/billing, and spawns GPU work on Modal. The heavy vision/LLM core lives
in ``src/`` and runs inside the Modal worker — never imported here at module load.
"""
