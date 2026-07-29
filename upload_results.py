"""Push a run's output directory to the Hugging Face Hub.

The training script mirrors milestone checkpoints as it goes, but the analysis artifacts
(feature_stability.csv, matching_info.npz, config.json, plots) are only written once at the
very end. On a leased machine that gets deleted on a deadline, those would not survive.
Run this after the analysis finishes -- or at any point, since it is idempotent.

Usage:
    export HF_TOKEN=hf_...
    export SAE_HF_REPO=you/your-repo
    export SAE_RESULTS_BASE=$HOME/sae-stability-outputs   # optional
    python upload_results.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

# Keep in step with prelim_experiments_update.py so both agree on where a run lives.
MODEL_NAME = "pythia-70m-deduped"
LAYER = 3
TOKEN_BUDGET_B = 8

# Uploaded last, so a partial upload still leaves the cheap, most-analysable files present.
LARGE_SUFFIXES = {".pt"}


def resolve_output_dir() -> Path:
    if os.environ.get("SAE_RESULTS_BASE"):
        base = os.environ["SAE_RESULTS_BASE"]
    elif Path("/content/drive/MyDrive").exists():
        base = "/content/drive/MyDrive/sae-stability-outputs"
    else:
        base = str(Path.home() / "sae-stability-outputs")
    return Path(base) / f"{MODEL_NAME}_L{LAYER}_{TOKEN_BUDGET_B}Btok"


def main() -> int:
    repo_id = os.environ.get("SAE_HF_REPO")
    if not repo_id:
        print("SAE_HF_REPO is not set; nothing to upload to.")
        return 1

    output_dir = resolve_output_dir()
    if not output_dir.exists():
        print(f"No run directory at {output_dir}")
        return 1

    files = sorted(p for p in output_dir.rglob("*") if p.is_file())
    if not files:
        print(f"{output_dir} is empty.")
        return 1

    # Small files first: if the upload is interrupted, the results worth having are already up.
    files.sort(key=lambda p: (p.suffix in LARGE_SUFFIXES, p.stat().st_size))

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"Uploading {len(files)} file(s), {total_mb:.0f} MB from {output_dir}")
    print(f"                    to hf.co/{repo_id}\n")

    failed = []
    for path in files:
        rel = path.relative_to(output_dir).as_posix()
        size_mb = path.stat().st_size / 1e6
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=rel,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  ok    {rel}  ({size_mb:.1f} MB)")
        except Exception as e:
            failed.append(rel)
            print(f"  FAIL  {rel}  ({size_mb:.1f} MB): {e}")

    print()
    if failed:
        print(f"{len(failed)} of {len(files)} file(s) failed; re-run to retry just those:")
        for rel in failed:
            print(f"  {rel}")
        return 1

    print(f"All {len(files)} file(s) are on hf.co/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
