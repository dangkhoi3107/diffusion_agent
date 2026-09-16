# Camouflage Latent Blending v2

Sinh ảnh "ẩn subject" không cần huấn luyện: **SD 1.5 + ControlNet Canny/SoftEdge + dual-branch latent blending**
(luận văn *Deep Learning Model for Camouflage Effect Generation*). Đây là bản viết lại sạch của
`notebookd71c549cfb.ipynb`: đã sửa bug, có config YAML, một shell runner, và một lệnh chạy trên Kaggle.

```
camo/
  config.py      config dạng dataclass <-> YAML, --set key=value, import config JSON cũ
  hints.py       Stage 1: Canny, broken edges, layout transform (khớp với canvas)
  hed.py         HED soft-edge (giống hệt controlnet_aux, không cần cài controlnet-aux)
  masks.py       Stage 2: soft mask M = clip(R(B(H_s)))
  preprocess.py  Stage 1-2 cho một cặp subject/background (cache HED)
  generator.py   Stage 3-5: diffusion 2 nhánh + mask-guided latent blending
  schedules.py   gate g_t, alpha_t, trọng số theo layer của ControlNet
  metrics.py     SSIM_bg, PSNR_bg, S_edge, hidden_score (CPU)
  clip_score.py  CLIP recognizability gain (tùy chọn)
  presets.py     config preset + bộ variant ablation (Fig 4.7–4.11, seeds, lưới), dùng chung cho UI và CLI
  runner.py      run / batch / compare-sweep / stop, lưu ảnh + CSV + comparison sheet + zip
  ui.py          Gradio: chọn ảnh bằng click, generate nhiều seed, compare, batch, history
  ui_canvas.py   canvas kéo-thả/xoay đồng bộ với slider layout
  cli.py         python -m camo <command>
configs/default.yaml   cấu hình cuối trong luận văn + ghi chú khoảng tham số nên chỉnh
run.sh                 entry point duy nhất (tự cài dependency)
kaggle/camo_kaggle.ipynb   notebook 1 cell cho Kaggle
scripts/pack_kaggle.sh     đóng gói code thành dataset Kaggle
tests/                 test CPU (hình học, mask, config, metrics) + smoke test sampler
docs/ROADMAP.md        gợi ý cải tiến và cách áp dụng agent
```

## Chạy trên Kaggle

1. Đóng gói code ở máy local: `bash scripts/pack_kaggle.sh`, sẽ tạo ra `dist/camo-code.zip`.
   Upload file này lên *Kaggle → Datasets → New Dataset* (tên gợi ý: `camo-code`).
   Nếu đã cấu hình Kaggle CLI thì có thể dùng: `KAGGLE_USERNAME=<user> bash scripts/pack_kaggle.sh --upload`.
2. Tạo notebook từ `kaggle/camo_kaggle.ipynb`. Bật *Accelerator: GPU* và *Internet: On*.
   Add input gồm dataset `camo-code` và dataset ảnh (mặc định: `dangkhoi3107/thesis`).
   Có thể thêm *Secrets → `HF_TOKEN`* để tải model nhanh hơn.
3. Sửa `MODE`/`SETTINGS` trong cell rồi Run. Lệnh thực chất chỉ là:

```python
!bash {CODE}/run.sh ui                        # mở Gradio, click để render
!bash {CODE}/run.sh sweep --preset structure   # hoặc chạy không cần UI
```

Kết quả nằm ở `/kaggle/working/outputs/<lệnh>_<thời gian>/`. Với batch hoặc sweep dài, nên dùng *Save Version → Save & Run All*.
Nếu session bị ngắt, chạy lại với `RUN_NAME=<tên cũ> SKIP_EXISTING=1` để tiếp tục từ chỗ dừng.

## Giao diện Gradio (mọi thứ đều click được)

Trên Kaggle, đặt `MODE = "ui"` rồi Run. Cell sẽ in ra link `https://xxxx.gradio.live`. Mở link đó và giữ cell chạy.

| Khu vực | Làm gì |
|---|---|
| **Config preset** (trên cùng bên trái) | Thesis default · Reproduce thesis (legacy) · Fast draft · Subtler/Stronger subject · Textured subject photo, rồi bấm *Apply preset* |
| **Panel trái** | Mọi tham số theo stage của luận văn, cùng upload ảnh và Load/Save config (YAML hoặc `ui_config_*.json` cũ) |
| **Thumbnail** (trên cùng bên phải) | Click để chọn subject/background (từ thư mục hoặc ảnh upload) |
| **Generate** + slider *images* | Render 1–8 seed liền nhau cho cặp đã chọn, kèm bảng metrics. **Stop** dừng sau bước denoise hiện tại |
| Tab **Layout canvas** | Kéo/thu phóng/xoay subject, đồng bộ 2 chiều với slider |
| Tab **Stages** | Xem canny, softedge, 3 bước tạo mask, overlay (không chạy diffusion, vài giây) |
| Tab **Compare / ablation** | Chọn preset (Fig 4.7–4.11, seeds, scheduler, lưới 12 tổ hợp) hoặc tự dựng lưới tối đa 3 tham số. Có thể sửa danh sách variant. Phạm vi: cặp đã chọn / subject × mọi nền / mọi subject × nền / tất cả. Kết quả: comparison sheet (hàng = cặp ảnh, cột = variant), bảng metrics trung bình, ZIP |
| Tab **Batch** | Cùng một setting cho nhiều cặp ảnh (4 phạm vi như trên), kèm gallery, bảng metrics, ZIP |
| Tab **History** | Xem các output gần nhất, zip toàn bộ `outputs/` để tải về |

Định dạng danh sách variant (mỗi dòng một variant, key không ghi thì lấy giá trị từ panel trái):
```
weak mask: mask.gamma=1.5, mask.floor=0.15
medium (current):
seed+1: sampling.seed+=1
```

## Chạy local

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # hoặc bản torch phù hợp
pip install opencv-python-headless -r requirements.txt
SUBJECTS=data/animals BACKGROUNDS=data/bg OUT_DIR=outputs bash run.sh ui
```

## Các lệnh

| Lệnh | Việc làm |
|---|---|
| `bash run.sh ui` | Gradio (link share): canvas layout, preview stage, generate, batch, save/load config |
| `bash run.sh run` | Chạy cặp đầu tiên (subject đầu × background đầu) |
| `bash run.sh batch` | Mọi subject × mọi background, xuất `metrics.csv`, `grid.png`, `.zip` |
| `bash run.sh sweep --preset structure` | Giống tab Compare nhưng chạy không cần UI (dùng được với Save & Run All). Preset: `structure mask alpha schedule layout bg_strength guidance seeds scheduler tuning` |
| `bash run.sh sweep --grid blend.alpha_end=0.4,0.55,0.7 --grid mask.gamma=0.4,0.6` | Lưới tự chọn, xuất `summary.csv` và `comparison.png` |
| `bash run.sh list-presets` | In danh sách config preset và variant preset |
| `bash run.sh preview` | Chỉ chạy hint + mask + layout, không diffusion (vài giây/ảnh, dùng để chỉnh mask/layout) |
| `bash run.sh show-config --set ...` | In config sau khi áp override |
| `bash run.sh import-legacy --legacy-json ui_config_x.json` | Chuyển config JSON của notebook cũ sang YAML |

Có 3 cách cấu hình (cách sau ghi đè cách trước): file YAML (`CONFIG=...`), biến môi trường
(`SUBJECTS`, `BACKGROUNDS`, `OUT_DIR`, `STEPS`, `SEED`, `RUN_NAME`, `SKIP_EXISTING`, `SHARE`), và `--set section.key=value`.
Nếu gõ sai tên key hoặc giá trị ngoài miền hợp lệ, chương trình sẽ báo lỗi ngay.

## Tham số cần chỉnh

Thứ tự ưu tiên khi tuning (từ nhạy nhất đến ít nhạy):

| Tham số | Mặc định | Khoảng nên quét | Ảnh hưởng |
|---|---|---|---|
| `blend.alpha_end` | 0.54 | 0.35 – 0.70 | Độ hiện của subject. Thấp thì subject biến mất, cao thì thành "sticker" |
| `mask.gamma` | 0.5 | 0.4 – 0.9 | Nhỏ thì mask rộng và mạnh hơn, subject rõ hơn |
| `sampling.bg_strength` | 0.52 | 0.40 – 0.65 | Số bước thực chạy = `steps × bg_strength`. Cao thì nền bị vẽ lại nhiều |
| `control.softedge_scale` | 1.5 | 0.8 – 1.6 | Độ bám hình dáng subject |
| `sampling.guidance_sub` | 2.3 | 1.5 – 5.0 | Độ đúng loài. Cao thì dễ bị tương phản mạnh |
| `mask.floor` / `mask.ceiling` | 0.05 / 0.90 | 0.0–0.2 / 0.7–1.0 | `floor` cao giúp cắt phần rò rỉ khi ảnh subject có nền nhiều texture |
| `blend.end_frac` | 0.56 | 0.4 – 0.8 | Thời điểm alpha đạt `alpha_end` |
| `control.end_frac` | 0.84 | 0.6 – 0.9 | Tắt ControlNet sớm thì subject hòa vào nền tốt hơn |
| `control.canny_scale` | 0.02 | 0 – 0.4 | Giá trị 0.02 gần như tắt Canny. Muốn ablation Canny thật thì dùng khoảng 0.2 |
| `layout.scale` | 1.0 | 0.5 – 1.0 | Kích thước subject (đã sửa bug đảo ngược) |
| `sampling.guidance_bg` | 1.2 | 1.0 – 1.5 | Đặt 1.0 để tắt CFG nhánh nền, chạy nhanh hơn |

Preset "Tuning grid" (tab Compare, hoặc `bash run.sh sweep`) quét 3 tham số đầu (12 tổ hợp).
Cách chọn: sắp xếp `summary.csv` (hoặc bảng trong tab Compare) theo `clip_text_gain` (subject nhận ra được),
giữ các tổ hợp có `SSIM_bg` cao (nền được giữ), rồi xem `comparison.png` để chọn bằng mắt.

## Output của mỗi cặp

`output.png`, `subject.png`, `background.png`, `canny.png`, `softedge.png`, `mask_1_smoothed.png`,
`mask_2_remapped.png`, `mask_3_final.png`, `mask_overlay.png`, `panel.png` (dải 6 ảnh, dùng được cho hình
trong luận văn), `result.json` (prompt, seed, thời gian, metrics), `config.yaml` (tái lập được đúng ảnh này).

## Metrics

| Metric | Ý nghĩa |
|---|---|
| `SSIM_bg`, `PSNR_bg` | Mức giữ nền, tính có trọng số `1 − M` (ngoài subject). Cao là tốt |
| `SSIM_subject` | Mức giống nền trong vùng mask. Thấp nghĩa là subject đã thay đổi vùng đó |
| `S_edge` | Độ trùng cạnh subject với cạnh output trong mask |
| `hidden_score` | `−|mật độ cạnh trong mask − ngoài mask|`. Càng gần 0 thì vùng subject càng lẫn vào nền |
| `clip_text_gain` | `cos(output, "a photo of a {animal}") − cos(background, cùng text)`. Dương nghĩa là nhận ra được subject |
| `clip_image_gain` | Như trên nhưng so với chính ảnh subject |

## Bug đã sửa so với notebook

| # | Bug | Hậu quả | Sửa |
|---|---|---|---|
| 1 | `transform_frame` truyền ma trận nghịch đảo vào `cv2.warpAffine` (không có cờ `WARP_INVERSE_MAP`) | `scale=0.5` cho subject **to gấp 2**, rotation ngược chiều canvas, offset bị scale/xoay theo. Hình 4.8 và 4.13 bị ảnh hưởng | Dùng ma trận thuận. Test so với công thức vẽ của canvas |
| 2 | Corrector của UniPC dựng lại sample từ `last_sample` | Latent đã blend đưa vào `scheduler_bg.step()` bị thay thế, nên coupling không chạy đúng công thức (3.2.5) | Tắt corrector cho nhánh nhận latent blend (giữ corrector cho nhánh không coupling). Có `sampling.legacy_unipc_corrector: true` để tái tạo hành vi cũ |
| 3 | Silhouette tính từ Canny **đã làm đứt nét** | Mask silhouette gần như rỗng, nên `hidden_score` và auto attention compensation vô nghĩa | Bỏ các mode cũ, metrics dựa trên soft mask |
| 4 | `SSIM_bg` nhân cả hai ảnh với `1 − mask` rồi tính SSIM toàn ảnh | Vùng subject bằng 0 ở cả hai ảnh nên luôn "giống nhau", SSIM bị thổi phồng | SSIM map có trọng số |
| 5 | HED lỗi thì âm thầm dùng Canny làm soft-edge | Kết quả sai mà không có cảnh báo | Báo lỗi rõ (hoặc `hints.softedge_fallback_to_canny: true`) |
| 6 | Tên file đưa thẳng vào prompt | "a hidden lizard 000d3a9260 silhouette" | Bỏ token chứa số: `lizard_000d3a9260` thành `lizard` |
| 7 | Slider layout bị ghi đè bởi state cũ của canvas lúc bấm Generate; handle xoay đi ngược chuột | Chỉnh slider sau khi kéo canvas không có tác dụng | Canvas đồng bộ 2 chiều, handle xoay theo chuột |
| 8 | Lãng phí tính toán: ControlNet chạy cả nửa uncond rồi nhân 0; probe ControlNet mỗi ảnh chỉ để đếm block; batch chạy HED 2 lần mỗi cặp; `cudnn.deterministic`; attention slicing | Chậm | Chỉ chạy ControlNet trên nửa cond, gộp 2 nhánh vào 1 lần gọi UNet (kết quả y hệt), cache HED |

Kiểm chứng: chạy trên cùng model và input, `legacy_unipc_corrector=true` cho ra ảnh **giống hệt pixel**
với vòng lặp của notebook cũ. HED native cho kết quả giống hệt `controlnet_aux` (sai khác 0).

**Tái tạo ảnh trong luận văn:** `bash run.sh import-legacy --legacy-json ui_config_xxx.json` sẽ tự đổi
layout sang đúng hình học mà notebook cũ đã render. Sau đó chạy với `--set sampling.legacy_unipc_corrector=true`.

## Hạn chế đã biết

Soft mask lấy từ HED của **toàn bộ ảnh subject**. Nếu ảnh subject có nền nhiều texture, mask rò ra khắp
khung hình (đo trên ảnh thử: 95% pixel có mask > 0.05), làm nhánh subject vẽ đè lên nền. Tạm thời có thể
tăng `mask.floor` hoặc dùng ảnh subject nền trơn. Cách sửa gốc xem trong `docs/ROADMAP.md`.

## Test

```bash
python tests/test_core.py            # CPU: layout/canvas, chuyển đổi config cũ, mask, schedule, metrics
python tests/test_generator_tiny.py  # CPU: sampler 2 nhánh với model tí hon (cần torch + diffusers)
```
# diffusion_agent
