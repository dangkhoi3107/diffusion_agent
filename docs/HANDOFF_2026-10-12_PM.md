> **Đã cũ, xem [HANDOFF_2026-10-13.md](HANDOFF_2026-10-13.md).** File này chỉ còn giá trị lịch sử.

# Bàn giao CamouBlend agents: chiều 12/10/2026 (đóng băng lúc ~16:15)

File này **thay thế** [HANDOFF_2026-10-12.md](HANDOFF_2026-10-12.md). Bản sáng vẫn đúng ở các phần hạ tầng (mục 2) và phát hiện nghiên cứu (mục 3).

Đọc hết file này trước khi chạy lệnh nào. Mọi lệnh chạy từ gốc repo `/home/dangkhoi/code/diffusion`.

---

## 0. Tóm tắt 30 giây

- **Đang chạy, không cần làm gì:**
  - Runner local `scripts/labelset.py run` (PID 257650, chạy bằng `nohup setsid`) đang render lần lượt **9 đợt** `label_b1` … `label_b9` trên Kaggle GPU.
  - Mỗi đợt 50 cặp × 16 config = 800 ảnh, khoảng 2,4 giờ. Tổng khoảng 22 giờ, **xong dự kiến khoảng 14–15h ngày 13/10**.
  - Đợt nào xong là tự tải về `outputs/label_bK/`.
- **Bộ dữ liệu gán nhãn v2 đã dựng xong:**
  - 90 nền thuộc khoảng 56 loại.
  - 252 subject chia theo **10 nhóm hình dáng**, cộng 4 mặt người.
  - 450 cặp, mỗi nền 5 nhóm hình dáng khác nhau.
- **Việc của người dùng:** gán nhãn khi đợt về, tại `http://localhost:8765`.
  - Label server **chưa chạy**. Phiên mới cần bật nó khi `outputs/label_b1/` xuất hiện (xem mục 6).
- **Sự cố đã sửa trong phiên:**
  - (a) Zip tải dở bị coi là xong.
  - (b) **Kernel gắn bản `camo-code` cũ** nên đợt 1 render lại lưới v1, mất khoảng 2 giờ GPU.
  - (c) Quy tắc chia test của nền thiên nhiên đưa trọn 2 loại vào test.
- **Kế tiếp (sau khi có nhãn):** Model 1 giai đoạn A (DINOv2 đóng băng + MLP, CORAL + broken), rồi tới Model 2 (Proposer). Hạn **05/11/2026**.

---

## 1. Yêu cầu của người dùng (đầy đủ)

### 1.1 Đề tài và ràng buộc

- CamouBlend: chèn con vật hoặc mặt người vào nền sao cho **ngụy trang** (diffusion blend). Agent tự tìm config render cho từng cặp subject/nền.
- **Cô hướng dẫn yêu cầu:**
  - Không dùng API bên thứ ba (OpenAI…). Mọi agent phải **tự train/finetune**, train đơn giản cũng được.
  - Hạn khoảng **05/11/2026**.
  - Cô đã duyệt kiến trúc agent.
- **Kiến trúc đã chốt: 2 model.** Xem [AGENT_PLAN.html](AGENT_PLAN.html), artifact https://claude.ai/artifact/ME399mdv99wpLyMEwiZqQ9.
  - **Model 1 (Observer)**:
    - DINOv2-small, sau đó thêm LoRA.
    - Đầu vào là bộ ba [render, nền, render − nền].
    - Head A: visibility, ordinal CORAL theo thứ tự missing < subtle < good < too_visible.
    - Head B: broken.
    - Head C: loài (Recognizer, **đã train xong**, xem `outputs/recognizer/`).
    - Baseline: CLIP linear probe; GPT vision cũ chỉ khớp 50%.
  - **Model 2 (Proposer)**:
    - GRU/MLP nhỏ chọn config.
    - State = lịch sử (config, nhãn) + embedding nền.
    - Behavior cloning từ oracle trên lưới đã gán nhãn, sau đó REINFORCE.
    - So với Rule / Random / Fixed bằng `scripts/replay_agents.py`.
    - Phương án dự phòng nếu cô muốn có LM: Qwen2.5-0.5B LoRA SFT.
- AGENT_PLAN.html **đã cũ** ở 3 điểm: lưới 10 → 15 config (+C00), số ảnh 3.000 → 7.200, subject theo nhóm hình dáng.

### 1.2 Cách làm việc (bắt buộc)

- Viết **tiếng Việt**.
- Claude tự build và tự chạy hết. Người dùng chỉ **gán nhãn** và **ra quyết định nghiên cứu**.
- Cần chỉ rõ bấm gì. Không giao thao tác trên giao diện Kaggle nếu CLI làm được. Ưu tiên lệnh một dòng hoặc mở sẵn trang.
- Credential (token Kaggle) do người dùng tự nhập. Không bao giờ đọc hoặc in token.
- Nhãn của người dùng (rater `tac_gia`, `outputs/labels.jsonl`) là **ground truth**. Ấn tượng của Claude không thay được nhãn.
- **Hỏi lại** khi có quyết định nghiên cứu hoặc tài nguyên (GPU, số ảnh phải gán). Không hỏi những việc có mặc định hợp lý.
- Người dùng muốn **báo khi xong**. Họ không muốn bị gián đoạn bởi các bước nhỏ.

### 1.3 Các quyết định người dùng đã chọn (theo thứ tự thời gian)

| Khi | Quyết định |
|---|---|
| 03/10 | Kiến trúc agent: Proposer + Observer, Python thuần (không LangGraph). Nền rất quyết định kết quả: cần nhiều loại nền. |
| 05/10 | Không API. 3 agent tự train → sau đó gộp thành 2 model (Recognizer thành head C của Model 1). |
| 12/10 sáng | Thêm nền thiên nhiên **thành đợt mới** (không thay DTD). Đủ 3 nguồn: vỏ cây; lá/đá/nước/gỗ; cỏ/rêu/cát/đất/sỏi. |
| 12/10 sáng | Lưới **5 alpha × 3 bg_strength** (alpha 0–0,2 × bg 0,40/0,48/0,55) + C00. |
| 12/10 sáng | Subject **tập trung động vật, chia theo hình dáng** chứ không theo loài ("mèo và chó gần giống, mèo và thằn lằn khác"). |
| 12/10 sáng | Gán nhãn được tới khoảng 3.000 ảnh, đồng ý tăng khi thêm đợt. **Hiện là 7.200, cần xác nhận lại (mục 7).** |
| 12/10 chiều | Cho tắt runner cũ còn sót (trùng với runner mới). |
| 12/10 chiều | Nhóm many_legs (0 ảnh): **tự cắt bằng SAM trên Kaggle** (không bỏ nhóm). |
| 12/10 chiều | wings_spread: **thêm bướm Leeds**. Rắn để 7 ảnh. |
| 12/10 chiều | Nhóm hình dáng giữ hẳn cho test (dáng lạ): **round_shelled** (ếch + rùa). |
| 12/10 chiều | "**Các loại nền cũng quan trọng không kém**": vừa **bù đủ DTD** vừa **thêm nền thiên nhiên mới**. Chọn đủ 4 nhóm: lá khô/thảm lá; tuyết/băng; đáy biển/san hô; đá tảng/vách đá + bùn. |
| 12/10 16:00 | Đóng băng, viết bàn giao, chuyển sang phiên khác. |

Các quyết định Claude tự đưa ra theo mặc định hợp lý (chưa hỏi). Người dùng có thể đổi:
- Loài lạ cho test là **Rabbit, Horse, Penguin**. Hằng số `TEST_SPECIES` trong `scripts/labelset.py`.
- Mặt người ở khoảng 1/3 số nền (6,7% số cặp), dùng 4 ảnh mặt cũ.
- Thứ tự ưu tiên khi bù DTD: zigzagged, grid, dotted, banded. Bỏ lined, studded, waffled vì gần giống striped, polka-dotted, grid.
- Cuốn chiếu không thấy chân → legless_elongated. Bọ rùa hình vòm → round_shelled (vào test).
- Dựng lại b1 bằng subject v2 (b1 phải render lại vì sự cố stale).

---

## 2. Hạ tầng Kaggle: bổ sung bài học mới

Bản sáng mục 2 vẫn đúng: tài khoản `dangkhoi3107`, token `~/.kaggle/access_token`, máy local không có torch/GPU/sklearn, không mở trang `/edit`, chỉ một kernel GPU `camo-pilot-run` một lúc.

Bài học mới hôm nay:

1. **Dataset `camo-code` mới tải lên chưa có hiệu lực ngay.**
   - Ngay sau `kaggle datasets version`, `datasets status` vẫn báo "ready" (của bản cũ), và kernel đẩy lên sẽ gắn **bản cũ**.
   - Lúc 13:49, kernel bắt đầu 5 giây sau khi tạo version, nên render lại `grid.yaml` v1 suốt 2 giờ.
   - **Đã sửa trong `scripts/kaggle_run.py`:**
     - `wait_dataset()`: tải từng file (`grid.yaml`, `<đợt>/inputs.yaml`) từ bản Kaggle đang phục vụ và so sha256, khớp mới đẩy kernel.
     - `--skip-upload` cũng được kiểm tra; nếu Kaggle chưa có bản đúng thì tự tải lên.
     - Notebook có thêm bước tự kiểm tra `EXPECT`: lệch thì `RuntimeError("STALE_CAMO_CODE")` trong vài giây, máy chờ 2 phút rồi đẩy lại (tối đa 5 lần).
2. **Zip tải dở:** `kaggle kernels output` có thể để lại file zip 0 byte hoặc bị cắt cụt khi mạng đứt. `kaggle_run.py` và `fetch_subjects.py` nay kiểm tra bằng `zipfile.is_zipfile`, xoá file hỏng rồi tải lại.
3. CLI có khi tải xong zip rồi vẫn treo vài phút mới thoát. Nếu zip đã hợp lệ thì giải nén tay được.
4. Job CPU mới (kernel dạng script, có `--push`):
   - `camo-fetch-subjects-extra`: `python3 scripts/fetch_subjects.py --push --extra`
   - `camo-fetch-backgrounds`: `python3 scripts/fetch_backgrounds.py --push`
   - Phần đẩy/chờ/tải dùng chung là `fetch_subjects.push_kernel()`.
5. **Quota GPU tuần này:**
   - Đã dùng khoảng 4,3 giờ: 1,8 giờ lưới cũ + 0,5 giờ quét thử + khoảng 2 giờ cho đợt stale.
   - 9 đợt mới cần khoảng 21,6 giờ, tổng khoảng **26/30 giờ**.
   - Nếu hết quota giữa chừng, kernel báo lỗi và runner dừng với `state=error`. Phải chờ tuần sau rồi chạy lại `labelset.py run`; runner sẽ tiếp từ đợt chưa tải.

---

## 3. Đang chạy lúc đóng băng

| Tiến trình | Ở đâu | Ghi chú |
|---|---|---|
| `python3 -u scripts/labelset.py run` | local, PID 257650, log `outputs/labelset_run.log` (từ dòng `=== … runner v2`) | Tải `camo-code` lên, chờ bản mới có hiệu lực, đẩy b1, chờ, tải về, rồi b2 … b9. Trạng thái ở `outputs/.labelset_status.json` (trang gán nhãn đọc file này). |
| Kernel `dangkhoi3107/camo-pilot-run` | Kaggle GPU T4 | Mỗi đợt một version mới đè lên version cũ. |

Kiểm tra nhanh:

```bash
python3 scripts/labelset.py status
```

```bash
tail -5 outputs/labelset_run.log
```

Nếu runner chết, ví dụ do máy khởi động lại: chạy lại lệnh dưới. Runner tiếp tục an toàn; khi trạng thái là `rendering` + `pushed`, nó chỉ chờ rồi tải, không đẩy lại.

```bash
nohup setsid python3 -u scripts/labelset.py run >> outputs/labelset_run.log 2>&1 &
```

---

## 4. Bộ dữ liệu gán nhãn v2 (`pilot/labelset/manifest.json`)

### 4.1 Lưới

- `pilot/labelset/grid.yaml`: C00 + 15 config.
  - alpha {0; 0,04; 0,08; 0,12; 0,2} × bg_strength {0,40; 0,48; 0,55}.
  - Lý do chọn: alpha 0,2–1,0 gần như không đổi ảnh.
- Lần quét thử (6 cặp): mức chênh trong vùng con vật so với C00 có median 7,2/255, 0% ảnh dưới 2/255.
- Kiểm tra trên đợt thật bằng `python3 scripts/grid_check.py outputs/label_b1`.

### 4.2 Subject (`pilot/labelset_subject_picks.json`, Claude chọn bằng mắt)

| Nhóm hình dáng | Ảnh | Ví dụ loài |
|---|---:|---|
| quadruped_compact | 36 | mèo, chó, cáo/chó rừng, thỏ, sóc, gấu trúc đỏ |
| quadruped_tall | 21 | ngựa, dê/ibex, hươu cao cổ, alpaca |
| quadruped_heavy | 18 | voi, gấu, cừu, bò/bò rừng |
| low_elongated_legged | 16 | thằn lằn, kỳ nhông, cá sấu, rái cá |
| legless_elongated | 10 | rắn (7), cuốn chiếu (3) |
| streamlined_fins | 22 | cá, cá mập, cá heo, cá voi |
| wings_spread | 19 | bướm Leeds (11), dơi (5), cú/vịt đang bay |
| bird_upright | 26 | cú, vẹt, chim cánh cụt, vịt, thiên nga |
| round_shelled (**test**) | 23 | ếch, rùa, bọ rùa hình vòm |
| many_legs | 61 | nhện, cua, bọ cạp, rết, bọ cánh cứng, côn trùng, tôm càng |

- Cộng thêm 18 subject v1 (S01–S18: mèo, thằn lằn, bướm, cá, 4 mặt người). Subject v2 có ID `T001`…
- Nguồn:
  - Open Images: mask có sẵn. Riêng many_legs cắt bằng **SAM ViT-B** theo box.
  - Leeds Butterfly.
  - Provenance json nằm cạnh mỗi ảnh.
- Lọc:
  - Loại 63 ảnh: mask vỡ, chỉ có đầu, có người cưỡi, ảnh vẽ hoặc 3D, màu UV, đồ ăn đã nấu.
  - Bỏ ảnh gốc trùng giữa các lớp (ví dụ A34_3 = A36_9).
  - Sửa nhãn loài sai của Open Images (gấu trúc đỏ trong lớp Bear, kỳ nhông trong lớp Lizard, cá hề trong lớp Goldfish).
- **Test subject:** cả nhóm `round_shelled` + loài Rabbit, Horse, Penguin (`TEST_SPECIES`).

### 4.3 Nền: 90 ảnh, khoảng 56 loại

| Nhóm | Ảnh | Nguồn |
|---|---:|---|
| 6 nhóm DTD (regular_pattern_2d, granular, relief_shading, fibrous_overlapping, directional_flow, low_detail_blotchy) | 60 | DTD |
| nature_plant (vỏ cây, lá, gỗ, cỏ, rêu, **lá khô**) | 12 | BarkVN-50, FMD, GTOS |
| nature_ground (đá, nước, cát, đất, sỏi, **tuyết/băng, đáy biển, đá tảng, bùn**) | 18 | FMD, GTOS, Open Images |

- **DTD:** `bg_dtd/` trước đó thiếu 10/47 loại (tải chưa đủ). Đã bổ sung đủ 120 ảnh × 9 loại.
  - `labelset.py fill-dtd` đổi 6 ảnh trùng loại: D55 zigzagged, D49 grid, D43 dotted, D37 banded (test), D58 braided, D59 grooved.
  - Nhóm regular_pattern_2d nay có 10 loại khác nhau.
- **Đợt b9 (D81–D90):** ảnh chọn ở `pilot/labelset_nature2_picks.json`; ứng viên ở `outputs/backgrounds_v2/sheet_*.png`.
  - Lá khô lấy từ GTOS (256 px, hơi mờ) vì lớp "Leaf" của Open Images toàn lá tươi.
- **Test nền:**
  - DTD: 2 nền mỗi nhóm.
  - Thiên nhiên: `nature_split()`, gồm rêu#1, sỏi#1, gỗ#2, cát#2, bùn#1, đáy biển#2. Mỗi loại test vẫn còn 1 ảnh ở train.
  - Quy tắc cũ "cứ nền thứ 5" đã đưa trọn rêu và sỏi vào test: **đã sửa**.

### 4.4 Cặp và đợt

- 450 cặp, 9 đợt × 50 cặp. Mỗi nền nhận 5 subject thuộc **5 nhóm hình dáng khác nhau**.
  - Chọn nhóm ít được dùng nhất trước, rồi tới ảnh ít dùng, rồi tới loài ít dùng.
  - Cứ 3 nền có 1 mặt người.
  - Kết quả: mỗi nhóm 42 cặp, mặt người 30 cặp.
- **Một cặp là test khi nền hoặc subject là test** (`manifest.subject_split_rule`).
- 7.200 ảnh render. Trang gán nhãn tự ẩn ảnh gần như trùng C00 (chênh < 2/255); khi train, các ảnh này thừa hưởng nhãn C00 (xem `replay_agents.grid_labels`).
- Hiển thị **mù**: thứ tự xáo trộn, ẩn tên config.

### 4.5 Dữ liệu cũ (không gán nhãn)

- `outputs/oldgrid_label_b1/` và `outputs/stale_label_b1_v1grid/`: hai bản render đợt 1 theo lưới v1 cũ (550 ảnh mỗi bản), đều vô dụng.
- `outputs/bgmap_v1/`: 188 nhãn, đã dùng cho phân tích nền.

---

## 5. Việc đã làm trong phiên chiều (file)

| File | Thay đổi |
|---|---|
| `scripts/kaggle_run.py` | Kiểm tra zip hợp lệ. `wait_dataset`/`dataset_current` (chờ bản `camo-code` mới có hiệu lực). Notebook tự kiểm tra `STALE_CAMO_CODE` rồi đẩy lại. |
| `scripts/fetch_subjects.py` | `--extra` (Leeds + SAM). `push_kernel()` dùng chung. Kiểm tra zip. |
| `scripts/fetch_backgrounds.py` (mới) | 9 loại DTD còn thiếu + ứng viên 5 loại nền thiên nhiên mới từ nhãn ảnh Open Images (cắt hình vuông đồng đều nhất). |
| `scripts/labelset.py` | `fill-dtd`, `build-nature2`, `rebuild-subjects [--release]`, `nature_split()`. Chép subject có `src`. |
| `scripts/grid_check.py` (mới) | Đo mức chênh của từng config so với C00 trong một đợt. |
| `tests/test_labelset.py` (mới) | 7 test. **40 test CPU đều qua** (`python3 -m pytest -q tests --ignore=tests/test_generator_tiny.py`). |
| `pilot/labelset_subject_picks.json`, `pilot/labelset_nature2_picks.json` (mới) | Danh sách chọn bằng mắt. |
| `pilot/labelset/` | Manifest + b1…b9 đã dựng lại. |
| `bg_dtd/` | +1.080 ảnh DTD (9 loại). |
| `outputs/subjects_v2/`, `outputs/subjects_v2b/`, `outputs/backgrounds_v2/` | Ứng viên đã tải về. |

**Chưa commit git gì.** Phần lớn code là untracked. Hỏi người dùng trước khi commit. `bg_dtd/`, `bg_nature_raw/` và `outputs/` không nên commit.

---

## 6. Việc tiếp theo (theo thứ tự)

1. **Khi `outputs/label_b1/` xuất hiện** (khoảng 18:30 ngày 12/10):
   - `python3 scripts/grid_check.py outputs/label_b1`. Kết quả phải giống lần quét thử: median trong vùng con vật khoảng 3–10, gần như không ảnh nào dưới 2/255.
   - Kiểm tra `pairs 50, renders 750` (50 × 15 config, chưa tính C00). Nếu là 500 (50 × 10 config của lưới v1), kernel vẫn chạy bản cũ: dừng runner và xem lại mục 2.1.
   - Bật trang gán nhãn bằng preview `label` trong `.claude/launch.json` (`python3 scripts/label_server.py --port 8765`). Báo người dùng mở `http://localhost:8765`.
   - Phím: 1 quá rõ, 2 đạt, 3 hơi chìm, 4 mất hẳn, 5 lỗi, 0 không chắc, Z quay lại.
2. Các đợt sau tự về khoảng mỗi 2,4 giờ. Trang gán nhãn tự nhận đợt mới (poll 30 giây).
3. **Model 1 giai đoạn A** (có thể viết code trước khi đủ nhãn, chạy CPU hoặc GPU trên Kaggle):
   - Trích đặc trưng DINOv2-small (+ CLIP làm baseline) cho [render, nền, diff], cache lại.
   - Head CORAL visibility + broken. Chia 5 fold theo cặp; test theo quy tắc mục 4.4.
   - Thước đo: độ khớp với nhãn người dùng (good/not, exact), so với GPT 50%.
   - Head C (loài) đã có ở `outputs/recognizer/`: đánh giá lại trên subject v2, gồm các loài mới và many_legs.
4. **Model 2:** dựng sân tập từ lưới đã gán nhãn đầy đủ, train BC rồi REINFORCE; so với Rule/Random/Fixed bằng `scripts/replay_agents.py`.
5. Model 1 giai đoạn B (LoRA, GPU). Đánh giá end-to-end. Cập nhật AGENT_PLAN.html. Viết báo cáo cho cô.

---

## 7. Cần người dùng quyết định hoặc lưu ý

- **Số ảnh cần gán nhãn: khoảng 7.200** (9 đợt), trước đây nói được khoảng 3.000.
  - Gợi ý nếu không đủ sức: gán trước các cặp **train** của 4–5 đợt đầu. Phiên mới có thể thêm bộ lọc "chỉ train" vào trang gán nhãn.
  - Hoặc bỏ bớt đợt (đặt `hold` cho đợt chưa render).
- **Quota GPU** khoảng 26/30 giờ tuần này: sát. Đợt cuối (b9, nền thiên nhiên mới) có thể sang tuần sau.
- Token Kaggle từng lộ trong ảnh chụp (MVP_STATUS 03/10): nên revoke và tạo token mới sau đợt render.
- Ảnh lá khô (GTOS, 256 px) hơi mờ so với các nền khác.
- Nền `D37 banded` là loại DTD chỉ xuất hiện trong test (không còn ô trùng loại để đổi).

---

## 8. Bộ nhớ Claude

Đã cập nhật `~/.claude/projects/-home-dangkhoi-code-diffusion/memory/`:
- `self-trained-agents-plan.md`: label set v2.
- `kaggle-dataset-race.md`: bài học stale camo-code.
- `user-working-style.md`: như cũ.

---

## 9. Cập nhật sáng 13/10 (~07:30)

- **Đêm 12/10:**
  - Máy bị **tắt** lúc 18:21 (shutdown, không phải ngủ) và bật lại lúc 21:33, nên runner chết.
  - Kernel đợt 1 đẩy lúc 16:13 bị **treo**, Kaggle cắt nó khoảng 02:30 mà không có output.
  - Quota GPU hầu như không bị trừ, có lẽ kernel chủ yếu nằm chờ. Theo trang Settings của Kaggle, quota tuần là **8/30 giờ** lúc 07:20 ngày 13/10.
- **Runner v3** (khởi động lúc 01:01, chạy dưới `systemd-inhibit --what=idle:sleep` để máy không tự ngủ):
  - `check_batch()`: mỗi đợt tải về phải đủ đúng các config của lưới cho mọi cặp, và ít nhất 95% trial ok. Nếu không, đợt đó bị cất sang `outputs/stale_<run>_<time>` rồi render lại.
  - Khi lỗi: thử tải lại (`--download-only`) trước, sau đó render lại, tối đa `BATCH_RETRIES = 2` lần. Chỉ khi vẫn lỗi mới dừng với `state=error`.
  - Kết quả: đợt 1 render lại lúc 02:34, xong 05:16, **800/800 ok**. `grid_check` cho median trong vùng con vật 7,4/255, 3% dưới 2/255. Đợt 2 đẩy lên lúc 05:21.
- **Trang gán nhãn** (`label_server.py`):
  - Bỏ qua các thư mục `stale_*` và `oldgrid_*` (lưới v1, không gán nhãn).
  - Đợt 1 hiện **744 ảnh**; 56 ảnh gần trùng C00 tự ẩn.
  - Server chạy bằng preview `label`. Nếu tắt, chạy lại bằng `python3 scripts/label_server.py --port 8765`.
- **Dự kiến các đợt về** (khoảng 2,6–2,7 giờ mỗi đợt):

  | Đợt | Giờ về |
  |---|---|
  | b2 | ~08:05 |
  | b3 | ~10:45 |
  | b4 | ~13:25 |
  | b5 | ~16:05 |
  | b6 | ~18:45 |
  | b7 | ~21:25 |
  | b8 | ~00:05 ngày 14/10 |
  | b9 | ~02:45 ngày 14/10 |

  Tổng 7.200 ảnh (dưới mức 10.000 người dùng đặt ra). Quota cần thêm khoảng 20 giờ, còn khoảng 22 giờ: vừa đủ.
- **Edge (Claude in Chrome):** đã kết nối, Kaggle đăng nhập bằng `dangkhoilearning3107@gmail.com`.
  - Xem quota ở https://www.kaggle.com/settings (mục "Kaggle GPU").
  - Log của version mới nhất ở `…/camo-pilot-run/log`.
  - Trang Kaggle tải chậm qua extension: dùng `find` hoặc chụp màn hình thay cho `get_page_text`.
  - Kernel chạy trên **GPU T4 ×2**, nhưng pilot chỉ dùng 1 GPU. Chia đợt ra 2 GPU sẽ nhanh gấp đôi, nhưng chưa cần vì tốc độ gán nhãn mới là nút thắt.
- **Trang checklist** https://claude.ai/artifact/QbD5iVTsnYXPZNGM6DFeQQ: ArtifactData báo "không tồn tại hoặc không có quyền", nên chưa tick được.
- **Test:** python hệ thống 49 qua, 1 bỏ qua. Venv `~/.venvs/camo-ml`: `test_observer` và `test_labelset` 16/16 qua.
- **Nếu máy tắt hoặc khởi động lại**, chạy lại runner (nó tự tiếp tục, không đẩy lại đợt đang render):

  ```bash
  nohup setsid systemd-inhibit --what=idle:sleep --who=camo-labelset --why="Kaggle label batches" --mode=block python3 -u scripts/labelset.py run >> outputs/labelset_run.log 2>&1 &
  ```

---

## 10. Cập nhật 13/10 ~07:40: lưới v2 làm con vật mất hẳn → quét thử lưới v3

- **Người dùng thấy ảnh đợt 1 gần như "mất hết con vật".**
  - Kiểm tra trên cả 50 cặp: kể cả ở config rõ nhất `a020_b55`, chỉ khoảng 5–8 cặp còn thấy được con vật.
  - Nguyên nhân: C00 ở đợt 1 trông giống bgmap_v1, nhưng khoảng 13/15 config của lưới v2 làm ảnh **mờ hơn C00** (alpha 0–0,12), và lưới không có config nào làm ảnh rõ hơn C00.
  - bgmap_v1 chỉ render C00: 69 đạt / 62 mất hẳn / 46 quá rõ / 9 hơi chìm trên 188 ảnh.
- **Quyết định của người dùng:**
  - Quét thử các nút tăng độ rõ trước, rồi Claude đề xuất lưới v3 để người dùng duyệt.
  - **Chưa gán nhãn đợt 1** (sẽ render lại theo v3).
- **Trạng thái các đợt:**
  - b3–b9 đang **HOLD**. b2 render nốt theo v2.
  - Bản manifest trước khi hold được lưu ở scratchpad. Muốn bỏ hold thì xoá trường `hold`, hoặc dựng lại bằng `rebuild-subjects --release`.
- **Đợt quét `sweep_v3`:**
  - Inputs: `pilot/labelset/sweep_v3` (lệnh mới `labelset.py sweep --name … --batch 1 --pairs …`, 8 cặp: 6 cặp khó + D03T151, D04S18).
  - Lưới quét: `pilot/labelset/sweep_v3_grid.yaml`, 12 config: C00, base, guidance_sub 3,5/5/7, layer_weights uniform (±g3,5), bg 0,60, g5+bg 0,55, scale 1,3, softedge 3,0, mask ceiling 1/gamma 0,3.
  - Pilot v3 đã cho thấy guidance_sub 5 → 3/3 quá rõ, và uniform → 4/4 quá rõ.
  - Một lệnh nền tự đẩy đợt quét khi runner dừng sau b2. Kết quả về `outputs/sweep_v3`, log ở `outputs/sweep_v3_run.log`.
  - Bước sau: dựng lưới v3 khoảng 15 config, C00 ở giữa, trải từ mất hẳn tới quá rõ. Ví dụ: mức mạnh của subject {alpha 0,04 / 0,12 / 0,2 / g3,5 / g5} × bg {0,40 / 0,48 / 0,55}. Sửa `ACTION_SPACE` trong `camo/agent.py` (Rule đang giả định các trục có thứ tự), dựng lại `grid.yaml` + b1…b9, rồi render lại.
- **Model 2 (Proposer) đã viết xong:**
  - Mô hình ở `camo/proposer.py`: GRU, h0 lấy từ đặc trưng nền đã chiếu **PCA 16 chiều** (không whitening) + dropout. Train BC (DAgger) rồi REINFORCE (`rl_lr` nhỏ). Có `NetPolicy(Policy)` để dùng với Controller.
  - Script train: `scripts/train_proposer.py`. Chạy CV gom nhóm theo nền, so với Rule / Random / Fixed / Oracle, rồi chấm test và lưu `outputs/proposer/`.
  - Cần ít nhất 10 cặp train có đủ nhãn cả 15 config.
  - Trên dữ liệu giả 768 chiều (tín hiệu hạng thấp): success 1,0 với 1,2–1,6 lượt render, Rule cần 3,0 lượt.
  - Không có PCA thì mạng học vẹt nhiễu: success 0,44.
  - Test: `tests/test_proposer.py`, 8 test (5 test cần torch, chạy bằng `~/.venvs/camo-ml/bin/python -m pytest`).
- **Toàn bộ test:**
  - python hệ thống: 53 qua, 5 bỏ qua (cần torch).
  - venv: `test_observer` + `test_proposer` + `test_labelset` 24/24 qua.
- **Seed (người dùng hỏi):** mọi ảnh của bộ nhãn tới giờ đều dùng `sampling.seed = 1234`, vì `pilot.pair_config` đặt `seed_mode = fixed`.
  - Đã thêm khoá tuỳ chọn `seed_per_pair: true` ở đầu file lưới (`pilot.pair_seed`): mỗi cặp một seed ổn định (sha256 của pair id), giữ nguyên qua mọi config của cặp. Mặc định tắt. Test ở `tests/test_pilot.py`.
  - **Bật nó trong lưới v3.**
  - Đợt quét có thêm 4 config dùng seed thứ hai (`*_s2`, seed 98765) để đo xem nhãn đổi bao nhiêu khi đổi seed.

---

## 11. Cập nhật 13/10 ~08:15: nhãn đợt 1–2, nền DTD, mất mạng

- **Người dùng đã gán 769 ảnh của b1 và 323 ảnh của b2** (lưới v2). Kết quả:
  - C00 của b1: 25 mất hẳn / 15 quá rõ / 5 hơi chìm / **4 đạt** / 1 lỗi. Các config v2 gần như toàn bộ mất hẳn.
  - Chỉ **10/50 cặp của b1 có ít nhất 1 config đạt**; b2 thì 0/49.
  - Nền quyết định nhãn: cùng một nền cho cùng nhãn với cả 5 con vật.
- **Nguyên nhân chính là nền DTD chọn ngẫu nhiên** (lệnh `build`): khoảng 15/60 ảnh không phải texture. Ví dụ D01 là ảnh con bọ xít, D12 có con mèo, D60 là tranh mặt người, cộng thêm túi xách, quả bí, rổ mây, cảnh có trời. bgmap_v1 dùng ảnh **đồng đều nhất** của mỗi loại, nên C00 có 37% đạt.
  - Mask không phải nguyên nhân: cùng loại mask viền HED, độ phủ tương đương.
- **Người dùng muốn "giữ output như cũ nhất có thể":** C00, mask và prompt giữ nguyên. Các nút mới (`mask.border_px`, `seed_per_pair`) mặc định tắt.
- **Lệnh mới:**
  - `labelset.py repick-dtd [--exclude …]`: mỗi nền DTD đổi thành ảnh đồng đều nhất cùng loại, không trùng ảnh bgmap. Bản chạy thử cho 58 ảnh đổi, đều là texture sạch. **Chưa áp dụng.**
  - `labelset.py rebatch`: mỗi đợt có 50 nền khác nhau (người dùng đã chọn).
  - `labelset.py sweep`: lấy một nhóm cặp con của một đợt để quét thử.
  - Hai lệnh `repick-dtd` và `rebatch` chỉ chạy khi mọi đợt đang hold và chưa render.
  - Vì b1 và b2 đã có ảnh, phải cất chúng sang tên `oldgrid_v2_label_bK` trước.
- **Đợt quét `sweep_v3`:** 18 config × 8 cặp = 144 ảnh. Đẩy lên Kaggle lúc 08:07; log ở `outputs/sweep_v3_run.log`.
  - Có thêm lưới an toàn `outputs/.sweep_v3_fallback.sh` (chạy bằng nohup): nếu lần tải đầu thất bại, nó tải lại bằng `--download-only`, chờ kernel xong và thử lại tới 12 giờ.
- **Chịu được mất mạng** (`kaggle_run.py`):
  - `wait_kernel()`: gọi status thất bại thì coi như kernel vẫn đang chạy.
  - Tải về thử lại theo thời gian (`--download-hours`, mặc định 12).
  - `--download-only` chờ kernel xong rồi mới tải.
  - Test ở `tests/test_kaggle_run.py`.
- **Việc tiếp theo:**
  1. Xem ảnh quét bằng mắt và chọn các nút tăng độ rõ mà không đổi phong cách ảnh.
  2. Dựng lưới v3 quanh C00; bật `seed_per_pair: true` trong `grid.yaml`; gửi người dùng duyệt.
  3. Cất b1, b2; chạy `repick-dtd`, `rebuild-subjects`, `rebatch`.
  4. Render thử vài cặp để chắc tỉ lệ đạt quay lại gần mức bgmap, rồi mới bỏ hold và chạy lại cả 9 đợt.

---

## 12. Cập nhật 13/10 ~09:45: lưới v3 + đợt thử

- **Máy được tắt lúc 08:27, bật lại 09:14.** Ảnh quét `sweep_v3` được tải sau đó (144/144). Ảnh để xem: `outputs/sweep_v3_review/sweep_A.png`, `sweep_B.png`.
- **Kết quả quét** (mức chênh trung vị trong vùng con vật so với C00):

  | Thay đổi | Mức chênh | Ghi chú |
  |---|---:|---|
  | Chỉ đổi seed | 20 | ngang với nút mạnh nhất |
  | lwu (layer_weights uniform) | 22 | con vật thành nét vẽ viền |
  | lwu + guidance 3,5 | 29 | |
  | guidance 5 + bg 0,55 | 14 | |
  | guidance 7 | 10 | |
  | bg 0,60 | 9 | |
  | scale 1,3 / softedge 3 | ~8 | |
  | guidance 3,5 / 5 | 4–7 | |
  | mask, maskwide, border16 | ~4–5 | **không khác base** |

- **Người dùng chọn lưới v3 "mức mạnh × bg, giữ phong cách"**, và chọn render thử 10 cặp trước.
  - `camo/agent.py`: `ACTION_SPACE = {subject.strength: a008, a020, g50, g70, lwu; sampling.bg_strength: 0.44, 0.52, 0.60}`. `STRENGTH` ánh xạ từng mức sang các ghi đè khi render. Hàm `render_overrides()`. `START = a020_b52`. Các id có dạng `g50_b60`.
  - Đã cập nhật Rule, LLM prompt, demo, proposer và toàn bộ test.
  - `grid.yaml` có `seed_per_pair: true`.
- **Bộ nhãn v3 (đã chạy thật):**
  - `labelset.py new-grid`: các đợt được đổi tên thành `label_b<k>v3`.
  - `label_b1` và `label_b2` (lưới v2, có nhãn) được đánh dấu bằng file `outputs/<run>/RETIRED`: không hiện trên trang gán nhãn nữa, nhưng nhãn vẫn được dùng để train.
  - `repick-dtd`: thay 58 nền DTD bằng ảnh đồng đều nhất cùng loại.
  - `rebatch --trial 10`: tạo **đợt thử `label_b0v3`** (10 cặp, 10 nền, 8 nhóm, không bị hold), và 9 đợt mỗi đợt 48–49 cặp với 48–49 nền khác nhau, đều đủ 8 nhóm. Các đợt này **hold**, chờ kết quả đợt thử.
- **Bản sao lưu trước khi đổi** nằm trong scratchpad (`backup_v2_*`).
- **Bước tiếp theo:**
  1. Đợt thử về. Người dùng gán nhãn (khoảng 160 ảnh).
  2. Đếm số cặp có ít nhất một ảnh "đạt". Mục tiêu: cao hơn rõ so với lưới v2 (b1: 10/50; b2: 0/49).
  3. Nếu đạt: bỏ hold đợt 1–9 (xoá trường `hold` trong manifest) rồi chạy `labelset.py run`. Cần khoảng 24 giờ GPU; quota đã dùng khoảng 12/30 giờ.
- **Test:** 64 qua, 5 bỏ qua (cần torch). Venv: observer + proposer + labelset + kaggle_run đều qua.
