"""Entrypoint chạy pipeline realtime: video -> depth + pose/track -> video annotated + skeleton_data.jsonl

Usage:
    python scripts/run.py --source 0                    # webcam
    python scripts/run.py --source path/to/video.mp4
    python scripts/run.py --source video.mp4 --config configs/default.yaml
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.depth_model import DepthModel
from src.pose_model import PoseTracker
from src.data_writer import SkeletonDataWriter
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pose", action="store_true", help="Chỉ chạy depth, bỏ qua pose/skeleton.")
    parser.add_argument("--source", required=True, help="Đường dẫn video, hoặc '0' cho webcam mặc định.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--display", action="store_true", help="Hiện cửa sổ xem trực tiếp (bắt buộc nên bật khi test camera thật).")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    paths = cfg["paths"]
    dcfg = cfg["depth"]
    pcfg = cfg["pose"]
    pipe_cfg = cfg["pipeline"]
    out_cfg = cfg["output"]

    # ---------- source: số -> webcam, chuỗi -> file ----------
    source = int(args.source) if args.source.isdigit() else args.source

    Path(out_cfg["dir"]).mkdir(parents=True, exist_ok=True)
    video_out_path = os.path.join(out_cfg["dir"], out_cfg["video_name"])
    data_out_path = os.path.join(out_cfg["dir"], out_cfg["data_name"])

    print("🔧 Đang nạp model depth (PyTorch, FP16)...")
    depth_model = DepthModel(
        repo_dir=paths["repo_dir"],
        checkpoint_path=paths["depth_torch_ckpt"],
        encoder=dcfg["encoder"],
        input_size=dcfg["input_size"],
        use_torch_compile=dcfg["use_torch_compile"],
        colormap=dcfg["colormap"],
        batch_size=dcfg["batch_size"],
    )

    pose_tracker = None
    if not args.no_pose:
        print("🔧 Đang nạp model pose + tracker...")
        pose_tracker = PoseTracker(
            weights_path=paths["pose_weights"],
            imgsz=pcfg["imgsz"],
            half=pcfg["half"],
            conf=pcfg["conf"],
            tracker=pcfg["tracker"],
        )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Không mở được nguồn video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(source, str) else 0

    if pipe_cfg["resize_input_to"]:
        width, height = pipe_cfg["resize_input_to"]

    video_writer = None
    if pipe_cfg["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height), isColor=True)

    data_writer = SkeletonDataWriter(data_out_path) if pipe_cfg["save_skeleton_data"] else None

    pipeline = Pipeline(
        depth_model=depth_model,
        pose_tracker=pose_tracker,
        data_writer=data_writer,
        queue_maxsize=pipe_cfg["queue_maxsize"],
        every_n_frames_pose=pcfg["every_n_frames"],
        resize_input_to=pipe_cfg["resize_input_to"],
    )

    print(f"🎬 Bắt đầu xử lý (nguồn video gốc: {fps:.1f} FPS)...")
    report = pipeline.run(cap, video_writer=video_writer, total_frames=total_frames, live_preview=args.display)

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if data_writer is not None:
        data_writer.close()

    print("\n================ BÁO CÁO HIỆU NĂNG ================")
    print(f"⏱️ Tổng thời gian: {report['total_time_s']:.2f}s cho {report['frames']} frames")
    print(f"📊 FPS thực tế: {report['fps_achieved']:.1f}  (nguồn: {fps:.1f})")
    print(f"   -> {'✅ ĐẠT realtime' if report['fps_achieved'] >= fps else '⚠️ CHƯA đạt realtime'}")
    print(f"⏱️ Depth trung bình: {report['depth_ms_avg']:.2f} ms/frame")
    print(f"⏱️ Pose  trung bình: {report['pose_ms_avg']:.2f} ms/frame")
    print(f"⏱️ Tổng suy luận GPU trung bình: {report['total_infer_ms_avg']:.2f} ms/frame")
    if pipe_cfg["save_video"]:
        print(f"✅ Video: {video_out_path}")
    if pipe_cfg["save_skeleton_data"]:
        print(f"✅ Skeleton data: {data_out_path}")


if __name__ == "__main__":
    main()
