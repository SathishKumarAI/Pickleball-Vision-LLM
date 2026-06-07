"""Modal GPU worker package.

`runner.run_job` is the pure orchestration of the analysis seam (idempotency,
download → analyze → upload, progress/cancel, usage), injectable and testable
offline. `modal_app` wires it to real Modal GPU + Supabase + the vision Pipeline.
"""
