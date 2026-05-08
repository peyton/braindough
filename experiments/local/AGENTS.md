# Local Experiments

- Specs in this directory may use large local models and hardware-specific
  acceleration.
- Keep model/cache/scratch/output paths under `BRAINDOUGH_HOME`.
- If a local backend cannot run, it should emit a skipped artifact with the
  blocker rather than hiding the failure.
