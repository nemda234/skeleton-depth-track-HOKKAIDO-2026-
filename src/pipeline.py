"""Kiến trúc 3 luồng: reader (đọc frame) -> infer (depth + pose/track trên GPU)
   -> writer (ghi video). Ghi data skeleton được làm ngay trong luồng infer vì
   json.dumps của vài chục số rất rẻ, không đáng tách luồng riêng.

   Lý do tách luồng: nếu chạy tuần tự, GPU phải rảnh trong lúc CPU đọc/ghi đĩa.
   Tách luồng giúp GPU luôn có việc để làm liên tục.
"""
import queue
import threading
import time

import cv2
import numpy as np
from tqdm import tqdm

STOP = None


class Pipeline:
    def __init__(self, depth_model, pose_tracker, data_writer,
                 queue_maxsize: int = 8, every_n_frames_pose: int = 1,
                 resize_input_to=None):
        self.depth_model = depth_model
        self.pose_tracker = pose_tracker   # có thể là None -> chỉ chạy depth
        self.data_writer = data_writer
        self.every_n_frames_pose = max(1, every_n_frames_pose)
        self.resize_input_to = resize_input_to

        self.frame_queue = queue.Queue(maxsize=queue_maxsize)
        self.result_queue = queue.Queue(maxsize=queue_maxsize)

        self.stage_times = {"depth_ms": [], "pose_ms": [], "total_ms": []}
        self._last_people = []  # giữ kết quả pose gần nhất để dùng lại ở các frame bị skip
        self.stop_event = threading.Event()  # camera không tự hết -> cần cờ để dừng thủ công (nhấn 'q')
    # ---------------- luồng 1: đọc video ----------------
    def _reader(self, cap):
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            if self.resize_input_to:
                frame = cv2.resize(frame, tuple(self.resize_input_to))
            self.frame_queue.put(frame)
        self.frame_queue.put(STOP)

    # ---------------- luồng 2: suy luận GPU ----------------
    def _infer(self):
        frame_idx = 0
        t_video_start = time.time()
        while True:
            frame = self.frame_queue.get()
            if frame is STOP:
                self.result_queue.put(STOP)
                return

            t0 = time.time()
            depth_map, depth_bgr = self.depth_model.infer(frame)  # batch_size=1, xem src/depth_model.py
            t1 = time.time()

            if self.pose_tracker is None:
                # chỉ chạy depth, không có pose
                people = []
                annotated = depth_bgr if depth_bgr is not None else frame
            else:
                run_pose = (frame_idx % self.every_n_frames_pose == 0)
                if run_pose:
                    people, pose_result = self.pose_tracker.track(frame, depth_map=depth_map)
                    self._last_people = people
                    annotated = pose_result.plot(img=depth_bgr, boxes=False, labels=False) \
                        if depth_bgr is not None else pose_result.plot(boxes=False, labels=False)
                else:
                    people = []
                    annotated = depth_bgr if depth_bgr is not None else frame
            t2 = time.time()

            self.stage_times["depth_ms"].append((t1 - t0) * 1000)
            self.stage_times["pose_ms"].append((t2 - t1) * 1000)
            self.stage_times["total_ms"].append((t2 - t0) * 1000)

            if self.data_writer is not None:
                ts_ms = (time.time() - t_video_start) * 1000
                self.data_writer.write_frame(frame_idx, ts_ms, people)

            self.result_queue.put(annotated)
            frame_idx += 1

    # ---------------- luồng 3: ghi video ----------------
    def _writer(self, out, pbar, live_preview: bool = False):
        while True:
            frame = self.result_queue.get()
            if frame is STOP:
                if live_preview:
                    cv2.destroyAllWindows()
                return
            if out is not None:
                out.write(frame)
            if live_preview:
                cv2.imshow("depth + skeleton (nhan 'q' de thoat)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
            pbar.update(1)

    def run(self, cap, video_writer=None, total_frames: int = 0, live_preview: bool = False):  # <-- thêm live_preview
        pbar = tqdm(total=total_frames if total_frames > 0 else None)

        t_read = threading.Thread(target=self._reader, args=(cap,), daemon=True)
        t_infer = threading.Thread(target=self._infer, daemon=True)
        t_write = threading.Thread(target=self._writer, args=(video_writer, pbar, live_preview), daemon=True)  # <-- thêm live_preview vào args

        t_start = time.time()
        t_read.start()
        t_infer.start()
        t_write.start()

        t_read.join()
        t_infer.join()
        t_write.join()
        pbar.close()

        total_time = time.time() - t_start
        n = len(self.stage_times["total_ms"])
        report = {
            "frames": n,
            "total_time_s": total_time,
            "fps_achieved": n / total_time if total_time > 0 else 0.0,
            "depth_ms_avg": float(np.mean(self.stage_times["depth_ms"])) if n else 0.0,
            "pose_ms_avg": float(np.mean(self.stage_times["pose_ms"])) if n else 0.0,
            "total_infer_ms_avg": float(np.mean(self.stage_times["total_ms"])) if n else 0.0,
        }
        return report
