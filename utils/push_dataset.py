#!/usr/bin/env python3
"""
Upload a locally-recorded LeRobot dataset to the Hugging Face Hub.

Uses LeRobotDataset.push_to_hub() so the dataset card and the CODEBASE_VERSION
tag are created too (a plain file upload would skip those, and lerobot needs the
version tag to load the dataset back from the Hub).

Your `hf` login must have WRITE access to the target repo_id's namespace
(user or org). Check with:  hf auth whoami

Usage:
    # push under the same name the folder was recorded as
    python utils/push_dataset.py --root dataset/samithva/bimanual_test --repo-id samithva/bimanual_test

    # keep it private
    python utils/push_dataset.py --root dataset/samithva/bimanual_test --repo-id samithva/bimanual_test --private
"""
import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="local dataset folder (contains meta/, data/, videos/)")
    ap.add_argument("--repo-id", required=True, help="target Hub repo, e.g. samithva/bimanual_test")
    ap.add_argument("--private", action="store_true", help="create the Hub repo as private")
    ap.add_argument("--no-videos", action="store_true", help="skip uploading videos/")
    args = ap.parse_args()

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    ds.push_to_hub(private=args.private, push_videos=not args.no_videos)
    print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
