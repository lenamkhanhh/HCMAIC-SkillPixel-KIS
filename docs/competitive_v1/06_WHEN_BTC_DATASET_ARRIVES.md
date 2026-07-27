# 06 — Khi BTC phát hành dataset

Không đổi model ngay. Làm theo thứ tự sau.

## 0–2 giờ: audit read-only

- xác nhận nguồn tải và rule sử dụng;
- tạo checksum/manifest;
- thống kê số video, duration, resolution, FPS/VFR;
- kiểm tra corrupt/missing file;
- kiểm tra audio, subtitle, metadata và language;
- xác nhận schema mapping và submission.

Không rename, transcode hoặc sửa dataset gốc.

## 2–4 giờ: adapter và validation

- tạo dataset adapter riêng;
- map field BTC sang canonical contract;
- chạy validator;
- xuất report lỗi theo video;
- kiểm tra timestamp bằng một sample có ground truth.

## 4–6 giờ: freeze baseline

- chọn 20–50 query hợp lệ, có nhiều loại difficulty;
- tạo qrels và review chéo;
- freeze hash của query/qrels/config/code;
- chạy incumbent không thay đổi;
- lưu per-query results và failure cases.

Sau baseline mới được chạy model hoặc fusion experiment.

## Điều kiện dừng

Dừng và báo lead nếu:

- rule cấm preprocessing/model/data đang định dùng;
- timestamp/schema không map chắc chắn;
- dataset có corruption đáng kể;
- query/qrels chưa được review;
- artifact không reproducible.
