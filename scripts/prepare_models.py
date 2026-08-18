"""Chạy 1 lần trước khi dùng pipeline: tải Depth Anything V2 checkpoint (PyTorch),
   clone repo gốc, và copy pose weights (.pt) đã có sẵn vào thư mục models/.

   Usage:
       python scripts/prepare_models.py --pose-weights /duong/dan/pose_model.pt
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pose-weights", required=True,
                         help="Đường dẫn tới file .pt YOLO-pose (bộ 26 keypoint) bạn đã có sẵn.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    paths = cfg["paths"]

    Path("models").mkdir(exist_ok=True)

    # 1. Clone repo Depth Anything V2 nếu chưa có (chứa code định nghĩa model)
    repo_dir = Path(paths["repo_dir"])
    if not repo_dir.exists():
        print("⏳ Đang clone Depth-Anything-V2...")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/DepthAnything/Depth-Anything-V2", str(repo_dir)],
            check=True,
        )
    else:
        print(f"✅ Repo đã có sẵn tại {repo_dir}")

    # 2. Tải checkpoint PyTorch (.pth) nếu chưa có
    ckpt_path = Path(paths["depth_torch_ckpt"])
    if not ckpt_path.exists():
        print("📥 Đang tải Depth Anything V2 checkpoint (vits)...")
        import urllib.request
        urllib.request.urlretrieve(paths["depth_checkpoint_url"], ckpt_path)
        print(f"✅ Đã tải về {ckpt_path}")
    else:
        print(f"✅ Checkpoint đã có sẵn tại {ckpt_path}")

    # 3. Copy pose weights người dùng cung cấp
    pose_src = Path(args.pose_weights)
    if not pose_src.exists():
        print(f"❌ Không tìm thấy file pose weights tại: {pose_src}", file=sys.stderr)
        sys.exit(1)
    pose_dst = Path(paths["pose_weights"])
    shutil.copy(pose_src, pose_dst)
    print(f"✅ Đã copy pose weights vào {pose_dst}")

    print("\n🎉 Chuẩn bị xong. Chạy tiếp: python scripts/run.py --source path/to/video.mp4")


if __name__ == "__main__":
    main()
