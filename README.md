# moerxiancheng-clj-xyj-proj

Research and task workspace for `clj-proj`, `xyj`, and related benchmarking or modeling scripts.

## Included

- Source code and task documents
- Config files and runnable scripts
- Charts and summaries that are suitable for version control
- LoRA-style training/inference validation summaries for MUSA runs

## Current Training Scope

- Training runtime tests use LoRA-style lightweight fine-tuning rather than full 8B parameter updates.
- The Llama3.1-8B backbone is loaded for real forward computation, while only low-rank adapter/probe parameters are updated.
- This keeps the tests aligned with the LoRA training requirement and avoids excessive full fine-tuning runtime.

## Excluded

- Local model weights
- Generated runtime artifacts and logs
- Python cache files
- Crash dump files
