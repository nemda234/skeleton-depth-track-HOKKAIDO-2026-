"""Wrapper cho Depth Anything V2 — PyTorch thuần + FP16 (khớp với cách notebook
   Kaggle thực tế đang dùng), KHÔNG dùng ONNX.

   Khác với bản batch=30 trên Kaggle (tối ưu THÔNG LƯỢNG tổng, chấp nhận độ trễ
   theo từng cụm frame), ở đây dùng batch nhỏ (mặc định 1) để tối ưu ĐỘ TRỄ —
   cần thiết vì mỗi frame còn phải chạy tiếp pose+tracking ngay sau đó, không
   thể đợi gom đủ 30 frame mới có kết quả.

   torch.compile mặc định TẮT trên local: trên GPU consumer (2050) thời gian
   warmup/recompile có thể mất vài phút và đôi khi lỗi tùy driver/kernel —
   không đáng đánh đổi cho video ngắn. Bật lại qua config nếu muốn thử.
"""
import sys

import cv2
import numpy as np
import torch


class DepthModel:
    def __init__(self, repo_dir: str, checkpoint_path: str, encoder: str = "vits",
                 input_size: int = 252, device: str = "cuda",
                 use_torch_compile: bool = False, colormap: str = "INFERNO",
                 batch_size: int = 1):
        sys.path.append(repo_dir)
        from depth_anything_v2.dpt import DepthAnythingV2  # noqa: import động sau khi append path

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        }
        self.device = device
        self.input_size = input_size
        self.batch_size = max(1, batch_size)
        self.colormap = getattr(cv2, f"COLORMAP_{colormap}") if colormap else None

        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model = model.to(device).eval().half()

        if use_torch_compile and hasattr(torch, "compile"):
            print("⚡ Đang biên soạn mô hình với torch.compile (có thể mất 1-2 phút warmup lần đầu)...")
            model = torch.compile(model)

        self.model = model

    @torch.no_grad()
    def infer_batch(self, frames_bgr: list):
        """Nhận list frame BGR, trả về list (depth_map_uint8, depth_bgr_color)."""
        orig_shapes = [f.shape[:2] for f in frames_bgr]

        batch = []
        for frame in frames_bgr:
            img = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            batch.append(img)

        tensor_batch = torch.from_numpy(np.array(batch)).to(self.device).half()
        raw = self.model(tensor_batch)
        raw = raw.squeeze(1).cpu().float().numpy() if raw.ndim == 4 else raw.cpu().float().numpy()

        results = []
        for i, (h, w) in enumerate(orig_shapes):
            d_map = cv2.resize(raw[i], (w, h), interpolation=cv2.INTER_LINEAR)
            d_map = cv2.normalize(d_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            d_bgr = cv2.applyColorMap(d_map, self.colormap) if self.colormap is not None else None
            results.append((d_map, d_bgr))
        return results

    def infer(self, frame_bgr: np.ndarray):
        """Tiện ích cho pipeline frame-by-frame (batch_size=1)."""
        return self.infer_batch([frame_bgr])[0]
