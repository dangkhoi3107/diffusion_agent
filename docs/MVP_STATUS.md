# CamouBlend MVP: trạng thái và phần chờ review (03-10-2026)

## Đã build và có test (32 test CPU pass)

| Thành phần | File | Kiểm chứng |
|---|---|---|
| Pilot M0/M1: chuẩn hoá input, preview, audit, manifest/resume, watchdog | `camo/pilot.py` | test + chạy thật v1–v4 trên Kaggle |
| Lấy input từ dataset công khai (Open Images, Leeds, AISegment, DTD) | `camo/sources.py` | chạy thật |
| Bản đồ 47 loại nền DTD × 4 con vật | `pilot/bgmap_inputs`, `scripts/bg_report.py` → `/backgrounds` | **xong 188/188 ảnh**, chờ chấm |
| Lưới v5: mặc định + 10 config agent × 18 cặp | `pilot/grid_v5.yaml` | đang chạy trên Kaggle (`grid_v5`, 198 ảnh) |
| Agent: action space, policy random/rule/LLM, controller có event log + resume | `camo/agent.py` | test |
| Observer VLM 2 lượt + few-shot + ảnh phóng to vùng subject; adapter OpenAI có streaming | `camo/llm.py` | **gọi API thật** (xem kết quả đo bên dưới) |
| Proposer LLM | `camo/agent.py` + `camo/llm.py` | **gọi API thật**: JSON hợp lệ, streaming, 1.7–3 s/lượt |
| Demo streaming 2 agent, ảnh TRƯỚC/SAU | `camo/demo.py` | chạy thật ở chế độ mô phỏng |
| Máy render GPU cho demo | `camo/render_server.py`, `kaggle_run.py --serve` | **chạy thật**: link gradio.live, render 29/29 bước stream về máy, 10.4 s/ảnh (đã sửa lỗi thư mục chỉ đọc + tải sẵn model) |
| Gán nhãn nhanh, tự lưu | `tools/label.html`, `scripts/label_server.py` | dùng thật (109 nhãn) |
| So sánh trước/sau bằng replay trên nhãn | `scripts/replay_agents.py` → `/compare` | test |
| Đo Observer khớp nhãn người | `scripts/eval_observer.py` | test với `--fake` |

## Observer so với nhãn tác giả (30 ảnh cùng mẫu, 03-10-2026)

| Cấu hình | Trùng hoàn toàn | Trùng đạt/không đạt | "Đạt" của bạn bị gọi "mất" |
|---|---:|---:|---:|
| gpt-5.4-mini, ảnh mặc định | 43% | 53% | 8 |
| gpt-5.4-mini, detail high + 5 ảnh mẫu | 23% | 57% | 4 |
| gpt-5.4-mini, detail high + phóng to vùng subject | n/a | 57% | 8 |
| gpt-5.5, detail high + phóng to | n/a | 50% | 1 (nhưng gọi "hơi chìm"/"lỗi") |

Kết luận: Observer **chưa đủ tin cậy để chạy tự động** (≈ đoán bừa ở mức đạt/không đạt). Model mạnh hơn thấy subject
nhưng tiêu chí khắt khe hơn tác giả → cần thêm nhãn nhất quán (ranh giới Đạt/Hơi chìm) để hiệu chỉnh; trong lúc đó demo
dùng chế độ human-in-the-loop (Observer chỉ gợi ý). Chi tiết: `outputs/observer_eval_*.json`.

## Cách chạy demo thật (render GPU Kaggle)

1. `python3 scripts/kaggle_run.py --user dangkhoi3107 --serve` (máy render tự tắt sau 40 phút không dùng).
2. Mở log Kaggle (`View Active Events` → camo-render-server → `Open Logs in Viewer`), copy dòng `RENDER_URL=https://….gradio.live`.
3. `python3 scripts/kaggle_run.py --serve-url https://….gradio.live && python3 -m camo.demo` → http://localhost:7861.

Kaggle CLI không đọc được log khi đang chạy nên bước 2 phải copy tay (hoặc nhờ Claude đọc qua trình duyệt).

## Chờ bạn review / làm

1. **Gán nhãn** trên http://localhost:8765: 188 ảnh `bgmap_v1` đã có; `grid_v5` tự xuất hiện khi về (ảnh trùng mặc định tự ẩn).
   Xem kết quả: http://localhost:8765/backgrounds (theo loại nền) và http://localhost:8765/compare (trước/sau agent).
2. **OpenAI API key** (platform.openai.com/api-keys) đặt vào biến môi trường `OPENAI_API_KEY` trên máy; sau đó:
   `python3 scripts/eval_observer.py --limit 30 --few-shot 1` và `python3 scripts/replay_agents.py --llm`.
3. **Quyết định cần duyệt** (khác PRD/tài liệu cũ, dựa trên nhãn pilot):
   - Action space đổi thành `blend.alpha_end` (profile constant: 0.2–1.0) × `sampling.bg_strength` (0.45/0.52);
     bỏ `alpha_end` dưới decay, `end_frac`, `softedge` vì gần như không đổi ảnh.
   - `layout.scale` giữ 1.0 (v1 14/18 đạt, 0.65 tệ hơn).
   - Không dùng LangGraph; controller Python thuần + event log.
   - Nền zigzag luôn thất bại: coi là giới hạn của phương pháp hay ca khó?
4. **Bảng audit C00–C06 trong GOLDEN_SET_GUIDE/MILESTONES** chưa cập nhật theo phát hiện trên.
5. Token Kaggle đã lộ trong ảnh chụp: revoke và tạo token mới sau khi xong.
