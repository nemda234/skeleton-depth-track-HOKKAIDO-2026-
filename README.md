# Realtime Video → Pseudo-3D Skeleton

Pipeline xử lý video để tạo **pseudo-3D skeleton** với thông tin `(x, y, depth)` theo từng `track_id`.

Hệ thống kết hợp:

* **Depth Anything V2** — ước lượng monocular relative depth
* **YOLO-Pose** — phát hiện người và keypoints
* **ByteTrack** — duy trì `track_id` giữa các frame
* **JSONL output** — lưu dữ liệu skeleton có cấu trúc để sử dụng cho các bước Action Recognition tiếp theo

Pipeline đồng thời xuất video trực quan để dễ kiểm tra kết quả tracking, pose và depth.

## 📁 Cấu trúc project

```text
skeleton-depth-track/
├── configs/
│   └── default.yaml          # Tham số tốc độ / độ chính xác
├── src/
│   ├── depth_model.py        # Depth Anything V2
│   ├── pose_model.py         # YOLO-Pose + ByteTrack
│   ├── data_writer.py        # Ghi skeleton + depth ra JSONL
│   └── pipeline.py           # Pipeline reader / infer / writer
├── scripts/
│   ├── prepare_models.py     # Chuẩn bị model
│   └── run.py                # Entrypoint
├── models/                   # Model đã chuẩn bị (gitignore)
├── outputs/                  # Output video + data (gitignore)
└── requirements.txt
```

## 🚀 Cài đặt

### 1. Tạo virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

Nếu PyTorch chưa nhận đúng CUDA, có thể cài phiên bản PyTorch phù hợp với hệ thống theo hướng dẫn chính thức:

https://pytorch.org/get-started/locally/

### 3. Kiểm tra GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu CUDA được nhận, kết quả sẽ có dạng:

```text
True NVIDIA ...
```

Nếu không có GPU CUDA, pipeline vẫn có thể chạy trên CPU nhưng tốc độ sẽ thấp hơn đáng kể.

## 🧠 Chuẩn bị model

Chạy:

```bash
python scripts/prepare_models.py --pose-weights /path/to/pose_model.pt
```

Trong đó `--pose-weights` là đường dẫn tới file YOLO-Pose `.pt`.

Script sẽ:

1. Clone repository **Depth Anything V2**
2. Tải checkpoint **Depth Anything V2 ViT-S**
3. Lưu checkpoint ở định dạng PyTorch `.pth`
4. Copy YOLO-Pose weights vào thư mục `models/`

Sau khi chuẩn bị:

```text
models/
├── depth_anything_v2_vits.pth
└── pose_model.pt
```

## ▶️ Chạy pipeline

### Webcam

```bash
python scripts/run.py --source 0
```

### Video

```bash
python scripts/run.py --source path/to/video.mp4
```

Pipeline sẽ đọc video, thực hiện depth estimation, pose estimation và tracking, sau đó ghi kết quả ra thư mục `outputs/`.

## 📦 Output

Sau khi chạy:

```text
outputs/
├── annotated.mp4
└── skeleton_data.jsonl
```

### `annotated.mp4`

Video trực quan bao gồm:

* Depth map
* Human skeleton
* `track_id`
* Thông tin tracking
* Các thông tin debug cần thiết để kiểm tra pipeline

### `skeleton_data.jsonl`

Mỗi dòng tương ứng với một frame và chứa dữ liệu của các person được tracking trong frame đó.

Các thông tin chính gồm:

* `frame_idx`
* `track_id`
* Tọa độ `(x, y)` của keypoint
* Confidence
* Depth tại từng keypoint

Dữ liệu có thể được sử dụng trực tiếp làm input cho các bước **Action Recognition**, ví dụ ST-GCN, mà không cần đọc và xử lý lại video gốc.

## ⚙️ Tối ưu hiệu năng

Pipeline được thiết kế để có thể điều chỉnh giữa **tốc độ** và **độ chính xác**, đặc biệt khi chạy trên các GPU có giới hạn VRAM.

Một cấu hình nhẹ có thể sử dụng:

```yaml
batch_size: 1

pose:
  imgsz: 256

depth:
  input_size: 252
```

Nếu gặp tình trạng chậm hoặc thiếu VRAM, có thể điều chỉnh theo thứ tự sau.

### 1. Giảm kích thước YOLO-Pose

```yaml
pose:
  imgsz: 224
```

Kích thước input nhỏ hơn giúp giảm thời gian inference và VRAM sử dụng.

### 2. Giảm kích thước Depth Anything V2

```yaml
depth:
  input_size: 224
```

`input_size` cần phù hợp với kiến trúc của model; với cấu hình hiện tại nên sử dụng kích thước chia hết cho `14`.

### 3. Chạy pose cách frame

```yaml
pose:
  every_n_frames: 2
```

Khi đó pose estimation chỉ chạy mỗi 2 frame, trong khi depth vẫn có thể được tính trên từng frame.

Cách này có thể giảm đáng kể chi phí inference của pose model.

### 4. Tắt `torch.compile`

```yaml
depth:
  use_torch_compile: false
```

`torch.compile` có thể giúp tăng tốc trong một số môi trường, nhưng cũng có thời gian warmup và phụ thuộc vào phiên bản PyTorch, CUDA và driver.

Với video ngắn hoặc môi trường thử nghiệm, có thể tắt tùy chọn này để giảm độ phức tạp.

### 5. Giảm resolution đầu vào

Nếu vẫn gặp vấn đề về hiệu năng hoặc VRAM, có thể giảm resolution của video thông qua:

```yaml
pipeline:
  resize_input_to: ...
```

Nên ưu tiên tìm mức resolution phù hợp với mục tiêu sử dụng thay vì cố đạt FPS cao nhất.

## 📈 FPS và mục tiêu của pipeline

Pipeline hướng tới việc **tạo dữ liệu skeleton chất lượng cho Action Recognition**, không nhất thiết phải đạt realtime 30 FPS trong mọi trường hợp.

Trong nhiều trường hợp:

> FPS ổn định và dữ liệu tracking liên tục quan trọng hơn FPS tối đa.

Pipeline cũng không sử dụng `time.sleep()` để cố mô phỏng tốc độ phát video gốc. Thay vào đó, hệ thống xử lý nhanh nhất có thể theo khả năng của phần cứng và cấu hình inference.

## 🔍 Tracking & Depth

### ByteTrack

Track ID được duy trì thông qua cơ chế tracking của Ultralytics:

```python
model.track(..., persist=True)
```

Việc duy trì `track_id` ổn định rất quan trọng đối với Action Recognition.

Ví dụ, skeleton của cùng một người cần tạo thành một chuỗi liên tục:

```text
Frame 1 → track_id 3
Frame 2 → track_id 3
Frame 3 → track_id 3
Frame 4 → track_id 3
...
```

Chuỗi này sau đó có thể được sử dụng làm temporal sequence cho các mô hình Action Recognition.

### Depth Anything V2

Depth Anything V2 cung cấp **monocular relative depth**.

Do đó:

```text
(x, y, depth)
```

nên được hiểu là **pseudo-3D coordinates**, không phải tọa độ 3D tuyệt đối trong không gian.

Depth không nên được sử dụng trực tiếp để suy ra khoảng cách thực tế theo mét nếu chưa có bước calibration hoặc metric depth phù hợp.

## 🎯 Pipeline tổng thể

```text
                    Video
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
      Depth Anything V2     YOLO-Pose
             │                 │
             │                 ▼
             │             ByteTrack
             │                 │
             └────────┬────────┘
                      ▼
              Skeleton + Depth
                      │
                      ▼
                  JSONL Data
                      │
                      ▼
              Action Recognition
                   (ST-GCN)
```

## 🔮 Mục tiêu mở rộng

Pipeline hiện tại tập trung vào việc tạo dữ liệu đầu vào cho Action Recognition.

Các bước tiếp theo có thể bao gồm:

* Chuẩn hóa skeleton sequence
* Xử lý missing keypoints
* Temporal interpolation
* Chuẩn hóa relative depth
* Xây dựng dataset train/validation/test
* Chuyển JSONL sang format phù hợp với ST-GCN
* Huấn luyện và đánh giá Action Recognition model
* So sánh 2D skeleton với pseudo-3D skeleton

## 📌 Lưu ý

Depth từ Depth Anything V2 là **relative depth**. Vì vậy, output `(x, y, depth)` nên được xem là representation pseudo-3D phục vụ machine learning, tracking và action recognition, thay vì hệ tọa độ 3D có đơn vị vật lý.

Mục tiêu chính của pipeline là biến video thành dữ liệu skeleton có **spatial information + temporal tracking**, tạo nền tảng cho các bài toán Action Recognition ở bước tiếp theo.
