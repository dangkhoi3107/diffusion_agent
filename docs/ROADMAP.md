# Hướng cải tiến và áp dụng agent

Thứ tự đề xuất: **(A) có số liệu → (B) sửa điểm yếu lớn nhất của phương pháp → (C) agent**.
Agent chỉ có ý nghĩa khi đã có metric để nó tối ưu và có một pipeline ổn định để nó điều khiển.

## A. Làm cho kết quả đo được (1–2 ngày GPU Kaggle)

1. Chạy `bash run.sh sweep` trên toàn bộ subject × background (12 tổ hợp mặc định). Kết quả là bảng
   `summary.csv` (hoặc tab Compare, preset "Tuning grid"), cho thấy vùng tham số tốt và độ nhạy của từng tham số. Bảng này thay thế được phần
   "chỉnh tay" trong mục 4.3 của luận văn.
2. Ablation định lượng cho Hình 4.9–4.11: `--grid control.canny_scale=0,0.2 --grid control.softedge_scale=0,1.5`,
   `--grid blend.profile=decay,ramp,constant`. Mỗi hình sẽ có thêm một bảng số đi kèm.
3. Thêm một **camouflage detector** (ví dụ SINet-V2 hoặc một mô hình COD pretrained) làm metric
   "độ khó phát hiện". Kết hợp với `clip_text_gain`, ta có 2 trục đối nghịch mà luận văn mô tả:
   *nhận ra được* và *khó phát hiện*. Vẽ Pareto front trên hai trục này.
4. Khảo sát người dùng nhỏ (20–30 ảnh, 2 câu hỏi: "thấy con gì?" và "ảnh có tự nhiên không?") để kiểm tra
   metric có tương quan với cảm nhận của người xem.

## B. Cải tiến phương pháp (theo tỉ lệ công sức / hiệu quả)

| Ý tưởng | Vì sao | Chỗ gắn vào code |
|---|---|---|
| **Segment subject** (BiRefNet / rembg / SAM2) rồi nhân vào soft mask | Sửa trực tiếp hạn chế đã đo được: HED lấy cả texture nền của ảnh subject nên mask rò khắp ảnh. Đồng thời xử lý được failure case "non-portrait / cluttered" | `preprocess.py`: `mask = mask * fg_alpha` trước `build_soft_mask`, thêm `mask.segment: true` |
| **Mask thích nghi theo cross-attention** của token `{animal}` (giống DiffEdit) | Mask cố định không biết mô hình đang "vẽ" subject ở đâu. Attention map cho vị trí thực tế theo từng bước | `generator.py`: hook attention processor của UNet nhánh subject, cập nhật `mask` mỗi k bước |
| **IP-Adapter** trên nhánh subject | Hiện nhánh subject chỉ có text + cạnh. Thêm ảnh subject làm điều kiện ngữ nghĩa giúp giữ đúng loài (failure case con mèo) | `CamoGenerator.load`: `sd.load_ip_adapter(...)`, truyền `image_embeds` cho nhánh subject |
| **Blend theo tần số**: chỉ trộn phần tần số thấp của latent subject | Ngụy trang = cấu trúc độ sáng tần số thấp, còn texture tần số cao nên giữ của nền. Giảm hiệu ứng "sticker" | Vòng lặp blend trong `generate()`: `fused = bg + w * lowpass(sub - bg)` |
| Base model tốt hơn (Realistic Vision SD1.5, SDXL + ControlNet Union) | Chất lượng texture nền | `models.base` (SD1.5 fine-tune đổi được ngay; SDXL cần thêm `add_time_ids`) |
| LoRA nhẹ trên COD10K | Future work 5 trong luận văn | Train riêng, load bằng `sd.load_lora_weights` |

## C. Áp dụng agent

Luận văn tự nêu hai hạn chế: *"outputs still require manual adjustment of mask strength, blending alpha,
guidance scale, seed, or layout"* và *"evaluation is mainly qualitative"*. Một agent dùng mô hình thị giác
(Claude Opus 5 / Sonnet 5 có vision) giải quyết đúng hai điểm này, và là đóng góp mới dễ bảo vệ.

### C1. Critic–Tuner agent (nên làm trước)

Vòng lặp: sinh ảnh, agent xem ảnh và metrics, đề xuất tham số mới, lặp lại.

```
state = config mặc định (hoặc điểm tốt nhất từ sweep)
lặp tối đa N=6 lần cho mỗi cặp subject/background:
    result = run_pair(session, cfg, subject, background, seed)          # tool có sẵn trong runner.py
    gửi cho LLM: [subject.png, background.png, panel.png], result.metrics,
                 các tham số hiện tại + khoảng cho phép (CHOICES, UNIT_INTERVAL_KEYS, bảng README)
    LLM trả JSON: {"scores": {"recognizable": 1-5, "natural": 1-5, "artifacts": 1-5},
                   "diagnosis": "...", "changes": {"blend.alpha_end": 0.48, "mask.gamma": 0.6}}
    validate(apply(changes))  → nếu điểm tổng không tăng sau 2 lần thì dừng
lưu lịch sử (ảnh, điểm, lý do) → dùng làm bảng "agent vs chỉnh tay vs sweep" trong luận văn
```

Các thiết kế quan trọng:
- **Không cho agent sửa code**. Nó chỉ gọi `set_value` trên whitelist key, `validate()` chặn giá trị sai.
  Cơ chế này đã có sẵn trong `config.py`.
- Agent đọc được "vì sao": bảng ảnh hưởng tham số trong README được đưa vào system prompt.
- Chấm điểm bằng LLM phải được đối chiếu với metric (CLIP gain, SSIM_bg) và khảo sát người dùng ở A4,
  để trả lời câu hỏi của hội đồng: "LLM chấm có đáng tin không?".
- Baseline so sánh: sweep lưới (A1), Optuna/TPE với cùng số lần gọi, và agent.
  Nếu agent đạt điểm tương đương với ít lần sinh ảnh hơn thì đó là kết quả chính.

### C2. Layout & prompt planner

Trước khi sinh ảnh, agent nhìn background và subject để quyết định: tên loài và prompt mô tả
(thay cho `subject_name` lấy từ tên file), cùng `layout.scale/offset/rotation`, tức là đặt subject vào vùng
texture phù hợp và tránh vùng quá rối. Cách này giải quyết failure case "layout quá đà" (Hình 4.13)
và biến bước kéo-thả canvas thành tự động.

### C3. Batch curator

Sau một batch lớn: agent duyệt `grid.png` và `metrics.csv`, đánh dấu failure (sticker, subject biến mất,
sai loài), gom nhóm theo nguyên nhân rồi viết báo cáo. Đây là cách nhanh để có mục "Failure cases" có số liệu.

### Hạ tầng

- Chạy agent trên Kaggle: API key để trong *Kaggle Secrets*, gọi API từ notebook, còn `Session` giữ model
  SD trên GPU giữa các vòng lặp nên không phải nạp lại.
- Mỗi vòng agent mất khoảng 1 lần sinh ảnh cộng 1 lần gọi API. Nên đặt ngân sách cố định cho mỗi cặp
  (ví dụ 6 vòng) và ghi log đầy đủ để tái lập.
