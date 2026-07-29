"""Upload the built dataset to the Hugging Face Hub.

The token is read from stdin, never from a file or an argument, so it does not
land on disk or in the process list of a shared host. Rotate it afterwards
regardless: a token that has been pasted anywhere should be considered spent.

The repository is created private. Visibility is a separate, deliberate step once
the upload is verified, because a half-uploaded dataset with a broken viewer is a
bad first impression that is hard to undo.
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. Helmcode/stack-v3-devops")
    parser.add_argument("--folder", default="dataset")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--public", action="store_true",
                        help="flip an existing repo to public; does not upload")
    parser.add_argument("--card-only", action="store_true",
                        help="upload just the dataset card, leaving the data alone")
    parser.add_argument("--message", default="Update dataset card")
    args = parser.parse_args()

    token = sys.stdin.read().strip()
    if not token:
        raise SystemExit("no token on stdin")

    api = HfApi(token=token)
    who = api.whoami()
    print(f"authenticated as {who.get('name')}", flush=True)

    if args.public:
        api.update_repo_settings(repo_id=args.repo, repo_type="dataset", private=False)
        print(f"{args.repo} is now public", flush=True)
        return

    readme = os.path.join(args.folder, "README.md")
    if not os.path.isfile(readme):
        raise SystemExit(f"missing dataset card at {readme}")

    if args.card_only:
        with open(readme) as handle:
            if "{{" in handle.read():
                raise SystemExit("card still has unrendered placeholders")
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=args.message,
        )
        print(f"card updated: https://huggingface.co/datasets/{args.repo}")
        return

    parquet = [
        os.path.join(root, name)
        for root, _, names in os.walk(os.path.join(args.folder, "data"))
        for name in names
        if name.endswith(".parquet")
    ]
    if not parquet:
        raise SystemExit(f"no parquet files under {args.folder}/data")
    total = sum(os.path.getsize(p) for p in parquet)
    print(f"uploading {len(parquet)} parquet files, {total / 1e9:.2f} GB, plus the card",
          flush=True)

    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)
    print(f"repo ready (private): https://huggingface.co/datasets/{args.repo}", flush=True)

    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=args.folder,
        allow_patterns=["*.parquet", "README.md"],
        num_workers=args.workers,
    )

    files = api.list_repo_files(repo_id=args.repo, repo_type="dataset")
    uploaded = [f for f in files if f.endswith(".parquet")]
    print(f"\nuploaded: {len(uploaded)} parquet files, card present: {'README.md' in files}")
    if len(uploaded) != len(parquet):
        raise SystemExit(
            f"expected {len(parquet)} parquet files on the Hub, found {len(uploaded)}"
        )
    print("verify the viewer, then flip to public with --public")


if __name__ == "__main__":
    main()
