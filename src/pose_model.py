"""Wrapper cho YOLO-pose (Ultralytics) + tracking (ByteTrack/BoT-SORT)
   + lấy mẫu depth tại từng khớp -> pseudo-3D skeleton."""
import numpy as np
from ultralytics import YOLO


class PoseTracker:
    def __init__(self, weights_path: str, imgsz: int = 320, half: bool = True,
                 conf: float = 0.4, tracker: str = "bytetrack.yaml", device: str = "cuda"):
        self.model = YOLO(weights_path)
        self.model.to(device)
        self.model.fuse()
        self.imgsz = imgsz
        self.half = half
        self.conf = conf
        self.tracker = tracker
        self.device = device

    def track(self, frame_bgr: np.ndarray, depth_map: np.ndarray | None = None):
        """Chạy detect+track trên 1 frame. PHẢI gọi tuần tự đúng thứ tự frame
        (persist=True dựa vào trạng thái nội bộ của tracker giữa các lần gọi).

        Trả về list các dict, mỗi dict là 1 người:
            {
                "track_id": int hoặc None (None nếu tracker chưa gán được ID),
                "bbox": [x1, y1, x2, y2],
                "keypoints_xy": [[x, y], ...],
                "keypoints_conf": [c, ...],
                "keypoints_depth": [d, ...] hoặc None nếu không truyền depth_map
            }
        """
        result = self.model.track(
            frame_bgr,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            conf=self.conf,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )[0]

        people = []
        if result.keypoints is None or result.boxes is None:
            return people, result

        boxes = result.boxes
        kpts_xy = result.keypoints.xy.cpu().numpy()          # [N, K, 2]
        kpts_conf = result.keypoints.conf
        kpts_conf = kpts_conf.cpu().numpy() if kpts_conf is not None else None  # [N, K]
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
        xyxy = boxes.xyxy.cpu().numpy()

        h, w = frame_bgr.shape[:2]
        for i in range(len(boxes)):
            xy = kpts_xy[i]
            conf = kpts_conf[i] if kpts_conf is not None else np.ones(len(xy))
            depth_vals = None
            if depth_map is not None:
                depth_vals = _sample_depth(depth_map, xy, w, h)

            people.append({
                "track_id": int(ids[i]) if ids[i] is not None else None,
                "bbox": xyxy[i].tolist(),
                "keypoints_xy": xy.tolist(),
                "keypoints_conf": conf.tolist(),
                "keypoints_depth": depth_vals.tolist() if depth_vals is not None else None,
            })
        return people, result


def _sample_depth(depth_map: np.ndarray, keypoints_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    xs = np.clip(keypoints_xy[:, 0].astype(int), 0, w - 1)
    ys = np.clip(keypoints_xy[:, 1].astype(int), 0, h - 1)
    return depth_map[ys, xs]
