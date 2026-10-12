# CamouBlend — Kế hoạch triển khai theo milestone và sprint

Ngày: 03-10-2026 · Phụ thuộc: [PRD v1.1](PRD_CAMOUBLEND.md).

Phạm vi đã chốt: animal + human portrait; portrait giữ hình dáng khuôn mặt và hòa nền, **không yêu cầu giữ danh tính**. Đây là backlog/acceptance plan, chưa phải báo cáo các tasks đã hoàn thành.

## 1. Cách tổ chức công việc

Một người có thể làm toàn bộ vai trò researcher/engineer/data curator theo thứ tự; người hướng dẫn và raters tham gia chốt rubric/đánh giá. Không cần tuyển đội nhiều agent để triển khai roadmap này.

Mỗi mốc phải có thư mục output hoặc báo cáo ngắn gồm: đã làm gì, inputs/versions, kết quả quan sát, failure còn lại, quyết định qua gate hay cần sửa. Không đánh dấu xong chỉ vì code chạy không lỗi.

Hai loại completion:

- **Technical done:** chức năng chạy đúng và có thể kiểm chứng.
- **Research outcome:** có lợi / chưa rõ / không có lợi. Kết quả “LLM không hơn rule” vẫn là outcome hợp lệ, không phải lý do vô hạn mở rộng hệ thống.

```text
M0 → M1 → M2 → M3 → M4 ───────────────→ M7 → M8
           └────────→ M5 → M6 optional ─┘
```

M5 có thể làm khi đã có labels ở M2; không đợi VLM mới đo metrics. M7 có thể đánh giá human-assisted MVP nếu M6 chưa đạt validation.

## 2. Lịch 12 tuần và việc làm từng bước

Planning assumption: một người phát triển khoảng 15–20 giờ/tuần, có GPU theo từng phiên và có người chấm tham gia ở giai đoạn cuối. Đây là dự toán thực hiện, không deadline đã cam kết. Nếu không có API hoặc raters, các nhánh phụ thuộc chưa hoàn thành; không lấp bằng kết quả giả.

| Tuần | Mốc / thứ tự công việc | Đầu ra cuối tuần | Phụ thuộc |
|---|---|---|---|
| 1 | M0: export config/env → normalize alpha/aspect → thu 6 subjects/6 backgrounds → QA | Baseline contract + input inventory + preview có alpha metadata | Code/GPU baseline |
| 2 | M1: audit 21 renders → chấm/kiểm tra leakage → repair preprocessing nếu cần → default 15 pairs | 36 output bundles hoặc feasibility report có nguyên nhân | M0 và inputs thật |
| 3 | M2/M3: rubric/form/eval ROI → manifest validator → event store → renderer wrapper | Annotation v1 + controller skeleton, chưa cần model calls | Audit outputs |
| 4 | M3: budget/resume/selector → random/rule/human → tests và 12-pair pilot | Baseline system chạy đúng, logs/cost so được | Contracts/evaluator |
| 5 | M4: GPT-5.4 mini proposer → strict JSON/repair → live HITL → history ablation | MVP human-assisted và preliminary comparisons | Baseline system + API preflight |
| 6 | M2 mở rộng/M5: dev120 coverage → 12-pair audit → labels/control metrics | Golden dev + diagnostic report v0 | Collection/raters; không cần full-auto |
| 7 | M6: cùng GPT-5.4 mini Observer A/B → development calibration → freeze policy | Vision pilot và measured call/token latency | Labels đủ hai domains |
| 8 | M5/M6: val30×3 configs + controls → release gate → study plan | Go auto hoặc giữ HITL; protocol và budget đã freeze | Validation tách dev |
| 9 | M7: test100 manifest freeze → chạy core3 methods/main seed → seed20 subset | Locked test run bundles, prefix selections, call/cost logs | Gate, test nguồn mới |
| 10 | M7: blinded final-output study + trajectory subset + manual comparator | Private annotations, no test tuning | Rater schedules/source exposure QA |
| 11 | M7: paired stats/domain breakdown/cost → failure analysis → targeted ablations đã đăng ký | Results tables, CI, supported/unsupported claims | Labels và complete accounting |
| 12 | M8: reproduce bundle → README → diagrams/slides → final review | Research package có thể review/tái lập | Kết quả đã đối chiếu logs |

Input collection có thể kéo dài qua tuần 3–8; không cần đợi dev120 để implement controller. Critical path gồm normalization/leakage, labels và rater availability, không chỉ code model adapter.

### 2.1. Mười ngày làm việc đầu tiên

| Ngày | Việc theo thứ tự | Bằng chứng hoàn thành |
|---|---|---|
| 1 | Đọc config hiện tại; chốt corrected UniPC/512/57/default; tạo run manifest mẫu | Export config/environment, không thay sampling |
| 2 | Thu/categorize sáu subjects; lưu originals; kiểm tra PNG alpha; chuẩn hóa RGB giữ tỷ lệ | O01–O06 + alpha/crop metadata, mặt không méo |
| 3 | Thu sáu backgrounds; tag representation/local texture; tạo pair IDs | B01–B06 hoặc gap ghi rõ; 18 dev rows |
| 4 | Preview ba audit pairs; QA hint/support/ROI/layout; fix input không hợp lệ | Debug panels + QA notes; chưa batch diffusion |
| 5 | Render C00–C06 cho ba pairs, lưu measured latency/config/output | 21 bundles hoặc errors có log |
| 6 | Review contact sheets; đánh dấu mask leakage, structure, sticker/hidden | Per-pair failure table và preprocessing go/no-go |
| 7 | Nếu leakage fail: alpha/support repair hoặc BiRefNet offline theo §6.1 PRD, rerun affected audit; nếu pass: baseline 15 pairs còn lại | Preprocess version freeze hoặc unresolved feasibility report |
| 8 | Chọn 4–6 ảnh neo nếu có; viết rubric animal/portrait, không identity | Rubric v1 và uncertain examples |
| 9 | Tạo annotation form và independent eval assets; thử chấm với người hướng dẫn | Raw labels, disagreements, ROI/version QA |
| 10 | Tổng hợp pilot; freeze action policy đầu; viết tasks/controller contracts cho M3 | Pilot findings + next sprint task IDs |

“Ngày” là đơn vị sắp việc, có thể tách thành nhiều buổi. Nếu ngày 7 phải repair preprocessing, dời downstream; không âm thầm bỏ QA để giữ lịch.

### 2.2. Cấu hình triển khai cố định

- Proposer và Observer: `gpt-5.4-mini-2026-03-17`, Responses API, reasoning none, strict JSON. Proposer text-only; Observer A/B tách context. Settings/caps ở PRD §17.
- Diffusion/HED/CLIP: giữ checkpoints trong current config; lock resolved revisions và runtime sau smoke.
- Segmentation có điều kiện: `ZhengPeng7/BiRefNet`, offline asset preparation, chỉ khi thiếu alpha/cutout tốt hoặc M1 cần repair. Không thêm segmentation vào agent loop.
- Data target: dev120/val30/test100, animal/portrait cân bằng ở từng split; pilot18 thuộc dev120, không phải một test set riêng.
- Core study: default + random + rule + proposed; ba search methods N=10, main seed1234; repeat20 pairs seed5678. Các ablations là phần thêm, không default chạy mọi tổ hợp.


## 3. M0 — Audit kỹ thuật và khóa contract

### Mục tiêu / input

Hiểu chính xác generator hiện tại, default và phần nào cần wrap. Input là code/config repo, GPU environment có sẵn và một vài input tạm để kiểm tra.

### Tasks

- [ ] M0.1 Ghi code revision, checkpoint IDs/revisions, dependencies, GPU/device/dtype.
- [ ] M0.2 Export full default config, phân biệt corrected và legacy scheduler behavior; chọn một baseline duy nhất cho nghiên cứu mới.
- [ ] M0.3 Xác nhận label/prompt formatting cho animal và portrait; lưu actual resolved prompt.
- [ ] M0.4 Xác định nơi đo preprocessing, render, metrics, serialization và end-to-end; cold load riêng warm runs.
- [ ] M0.5 QA `run_pair` artifacts và `Session` reuse; lập map fields hiện có vs cần thêm.
- [ ] M0.6 Chốt RunContract/manifest sơ bộ: immutable fields, source hashes, prompt/ROI versions, seeds.
- [ ] M0.7 Ghi các risks có thật: HED mask leakage, cache stale khi overwrite input, generation mask scoring, Cartesian batch ngoài ý muốn.
- [ ] M0.8 Implement/prepare normalization ngoài generator: alpha composite RGB128, preserve original alpha, aspect-preserving crop/pad 512, version/hashes.
- [ ] M0.9 Forward `Generation.steps_run` vào result metadata, log timing categories; measured fields không do LLM điền.
- [ ] M0.10 Lock installed package versions sau GPU smoke; current lower-bound requirements chưa phải runtime lock.

### Implementation tối thiểu

Thiết kế `schema.py`, `manifest.py`, `renderer.py`; chỉ implement adapter/timer tối thiểu khi bắt đầu code. Không sửa core sampling để chuẩn bị agent. Input files immutable hoặc versioned.

### Deliverables

- `baseline_contract.json` hoặc YAML tương đương, có version.
- `system_audit.md`: existing vs planned modules, environment, latency mẫu và known issues.
- Một output bundle có full config, prompt, seed, debug maps và thời gian đo thực.

### Gate

FR01–FR04 và FR18: baseline render được, không nhầm prompt/domain, biết nơi lưu artifacts và biết thời gian nào chưa đo. Nếu GPU/model unavailable, hoàn tất contracts/CPU validation và ghi rõ GPU smoke còn pending; chưa tuyên bố render đã kiểm chứng.

### Failure và xử lý

Nếu prompt portrait tạo sai nội dung, thử template trên dev và version thay đổi. Nếu mask lấy cả nền subject, QA inputs/preprocessing trước; không gán lỗi đó cho config optimizer.

## 4. M1 — Thu input và tạo những outputs đầu tiên

### Mục tiêu / input

Không cần có bộ ảnh mẫu sẵn. Thu sáu subjects/sáu backgrounds theo [input plan](INPUT_SELECTION_PLAN.md), rồi tạo outputs dùng xây rubric.

### Tasks

- [ ] M1.1 Thu O01 mèo, O02 thằn lằn, O03 bướm, O04 cá; hình dáng rõ, không mất bộ phận quan trọng.
- [ ] M1.2 Thu O05 portrait chính diện, O06 portrait góc 3/4 của người khác; đủ chi tiết mặt và nền tương đối sạch.
- [ ] M1.3 Thu B01 tường, B02 cát, B03 lá dày, B04 gỗ, B05 đá render 3D, B06 pattern 2D; nếu thiếu B05 ghi coverage gap.
- [ ] M1.4 Ghi original paths/hashes/source groups/provenance/style; không suy nhãn render 3D từ cảm giác chiều sâu.
- [ ] M1.5 Tạo đúng 18 pair IDs theo ma trận, tất cả là dev.
- [ ] M1.6 Preview/QA P02 mèo–cát, P13 portrait–tường, P16 portrait–lá; freeze layout.
- [ ] M1.7 Render 7 configs cho mỗi pair; cùng seed và frozen settings trong pair.
- [ ] M1.8 Lưu tất cả 21 outputs; chọn examples đạt nếu có, sticker, hidden, structure error, uncertain.
- [ ] M1.9 Nếu runner/QA ổn, chạy default 15 pairs còn lại; tổng 36 outputs.
- [ ] M1.10 Với PNG transparent, verify composite rõ màu và source alpha còn được lưu; không convert RGB trực tiếp rồi tin background sạch.
- [ ] M1.11 Nếu mask rò: dùng alpha/cutout support có sẵn; nếu thiếu thì BiRefNet offline + manual QA theo PRD §6.1. Log outside-mass/active diagnostics, không lấy whole-frame 95% làm threshold chung.
- [ ] M1.12 Nếu sửa preprocessing, freeze v2 cho mọi method và rerun affected outputs; giữ v1 artifacts riêng, không gộp số.
- [ ] M1.13 Chỉ qua batch gate khi hints/support/layout thẳng hàng và không cắt bộ phận quan trọng; failure chưa giải quyết phải có status.

### Bảng audit configs

| ID | bg_strength | alpha_end | softedge_scale |
|---|---:|---:|---:|
| C00 | 0.52 | 0.54 | 1.50 |
| C01 | 0.47 | 0.54 | 1.50 |
| C02 | 0.57 | 0.54 | 1.50 |
| C03 | 0.52 | 0.48 | 1.50 |
| C04 | 0.52 | 0.60 | 1.50 |
| C05 | 0.52 | 0.54 | 1.30 |
| C06 | 0.52 | 0.54 | 1.60 |

Mức SoftEdge không đối xứng; đây là sensitivity inspection, không phải derivative estimate. Không mở thêm tham số giữa chừng để làm ảnh đẹp mà không ghi separate experiment.

### Deliverables

Manifests, QA previews, 21→36 output bundles, contact sheets dùng diagnosis, `pilot_findings.md` mô tả từng domain và baseline runtime thực.

### Gate

Có inputs hợp lệ và outputs/logs có thể tái lập, biết failures chính. Không bắt mọi pair phải có success. Tuy nhiên, trước khi claim optimizer giải được cả hai domains, cần thấy generator có phạm vi tạo ảnh chấp nhận được ở cả hai; nếu một domain chưa có, ghi feasibility gap và audit thêm.

### Failure và xử lý

Nếu mọi portrait đều mất/méo mặt, kiểm tra style/preprocessing/prompt và miền tham số trên dev. Không bù bằng identity model vì identity nằm ngoài scope. Không drop khó âm thầm khỏi future test.

## 5. M2 — Golden set, rubric và human baseline v0

### Mục tiêu / input

Biến outputs thành feedback dùng được. Input: M1 outputs + inventory để mở rộng, không phải metric thresholds tự đặt.

### Tasks

- [ ] M2.1 Viết rubric common + domain-specific; portrait chấm face shape/structure, không identity.
- [ ] M2.2 Chọn ảnh neo từ dev; nếu chưa có accept, giữ best-available với lý do chưa đạt.
- [ ] M2.3 Chốt granularity animal labels và first-view protocol; portrait chỉ nhận ra mặt người.
- [ ] M2.4 Tạo annotation records accept/reject/uncertain, ratings, failure tags và pairwise/tie.
- [ ] M2.5 Tạo fixed expected ROI/background region, QA alignment; không lấy mask generation làm ground truth.
- [ ] M2.6 Thu manual trajectories trên ít nhất vài pair ở mỗi domain nếu có inputs phù hợp: hypothesis trước action, parent/config/seed, output, decision và timer.
- [ ] M2.7 Review disagreements với người hướng dẫn; sửa rubric trên dev, không rewrite raw labels cũ.
- [ ] M2.8 Mở inventory hướng tới khoảng 120 dev pairs, cân bằng domains; dùng selected pairing và nhiều nguồn mỗi nhóm.
- [ ] M2.9 Reserve validation/test sources mới; check shared image/person/background groups trước thử phương pháp.

### Implementation

`evaluation/schema.py`, `assets.py`, `human.py`; form chấm đơn giản trong Gradio hoặc collector riêng; JSONL raw annotation và private blind mapping. Chưa cần VLM, embedding hoặc vector DB.

### Deliverables

`rubric_v1.md`, `annotation_protocol_v1.json`, `pairs.jsonl`, fixed evaluation assets, raw annotations và `human_baseline_v0` trajectories. Coverage table đếm nguồn và pair riêng, không chỉ tổng outputs.

### Gate

Rater có thể dùng rubric và ghi uncertain khi cần; decision rules được giải thích bằng examples. Không đặt một agreement threshold tùy tiện chỉ để pass. Ghi agreement/disagreements và chốt tolerance phù hợp claim trước test. Labels và metrics tách rõ; file missing/error không bị coi là reject image.

### Failure và xử lý

Nếu disagreement về “hòa nền đẹp” lớn, bổ sung anchors/domain examples và tránh scalar hóa sớm. Nếu source groups không đủ độc lập, giảm scope hoặc thu thêm, không làm random split giả độc lập.

## 6. M3 — Controller và baselines trước LLM

### Mục tiêu / input

Có hạ tầng so sánh đúng và baseline mạnh. Input: validated pairs, rubric, allowed values và run contract.

### Tasks

- [ ] M3.1 Implement three-key policy, one-level actions, rebase và immutable fields.
- [ ] M3.2 Implement manifest-driven RendererAdapter quanh `run_pair`, Session reuse và timers.
- [ ] M3.3 Implement event history/snapshot, statuses, physical attempt count và artifact verification.
- [ ] M3.4 Implement bounded controller + human pending/resume + selector.
- [ ] M3.5 Implement fixed default, random global, ordered subset và rule/coordinate policy.
- [ ] M3.6 Rule mapping ghi là heuristic được kiểm tra dev, không guarantee monotonic; xung đột tags có ưu tiên rõ.
- [ ] M3.7 UI chấm/compare incumbent và controlled human tuning cùng bounds; tách unrestricted manual experimentation.
- [ ] M3.8 Tests budget/invalid actions/resume/failure/ROI/selection.
- [ ] M3.9 Chạy pilot N=5 trên cả hai domains, cùng seed policy; audit cost và outputs.

### Example rule baseline cần kiểm tra

```text
Nếu structure/presence fail:
    thử một action tăng khả năng bám cấu trúc đã có evidence trên dev
Nếu structure hợp lệ nhưng subject quá nổi:
    thử action giảm influence đã kiểm tra
Nếu background damaged:
    ưu tiên action kiểm soát noise/steps trong allowed policy
Nếu tag xung đột hoặc uncertainty cao:
    hỏi human / thử next legal coordinate theo lịch đã freeze
```

Rule phải ghi cụ thể action priorities sau M1/M2; đoạn trên chỉ mô tả logic. Không dùng rule biết config tốt nhất từ test.

### Deliverables

Controller package, baseline configs, functional tests, pilot run bundles, comparison table và một export có thể resume/review ở session khác.

### Gate

FR05–FR10/FR16: không action ngoài whitelist; budget accounting đúng; final pending trial chấm được khi hết render slots; selector không báo accept giả; baselines có cùng run contract. Functional done không đòi baseline thắng generator ở mọi pair.

### Failure và xử lý

Nếu rule đã gần đạt nhu cầu, vẫn giữ nó làm đối thủ của M4. Nếu lỗi hạ tầng chiếm nhiều runs, sửa trước khi kết luận search policy tốt/xấu.

## 7. M4 — Một LLM Config Proposer, hoàn thành MVP

### Mục tiêu / input

Kiểm tra giá trị của history-guided proposal. Input là logs có feedback từ M2/M3 và action policy freeze trên dev.

### Tasks

- [ ] M4.1 Tạo Responses API adapter cho snapshot `gpt-5.4-mini-2026-03-17`; strict JSON, reasoning none, token/latency/model/request-ID logging. Preflight truy cập/billing trước calls thật.
- [ ] M4.2 Prompt gồm task/rubric, immutable contract, evaluated history, allowed actions và budget còn lại.
- [ ] M4.3 Output schema: parent, key/direction, evidence IDs, summary và hypothesis ngắn.
- [ ] M4.4 Validator không tin config/cost/metric do model tự nói; bounded repairs/transport retry.
- [ ] M4.5 Replay một số dev histories để kiểm tra format/invalid actions; không dùng replay như proof search performance.
- [ ] M4.6 Chạy live LLM+human N=5 rồi N=10 khi ổn; cùng setup với rule/human/random.
- [ ] M4.7 Ablation no-history: giới hạn cùng budget/feedback đã định; report chính xác phần context bị bỏ.
- [ ] M4.8 Đánh giá quality, human active time và total cost; giữ snapshot đã chốt, freeze prompt/schema/policy versions. Không chọn lại model giữa test.

### Deliverables

MVP một Proposer, model-call logs, valid-action report, live trajectories, `mvp_results.md` nêu lợi ích hoặc null result. Có UI export và instructions thật, không chỉ pseudocode.

### Gate

Technical: proposals bounded/valid, history và calls truy vết đầy đủ. Research: có so sánh công bằng đủ để quyết định tiếp tục LLM hay ưu tiên rule. Không biến “model trả JSON đúng” thành bằng chứng agent hữu ích.

### Failure và xử lý

Nếu LLM không hơn rule, phân tích evidence/action errors và giữ kết quả. Có thể dừng ở rule-assisted product; nếu tiếp tục visual reasoning, nêu giả thuyết cụ thể rằng feedback thiếu thông tin, không thêm calls vô mục đích.

**MVP release có thể kết thúc sau M4:** human-assisted system hoạt động, biết giới hạn và có baseline comparison. Chưa claim tự động đánh giá ảnh.

## 8. M5 — Kiểm định metrics và evaluator policy

### Mục tiêu / input

Xác định signal nào phản ánh mục tiêu và signal nào dễ bị đánh lừa. Input: annotated dev/validation outputs và controls.

### Tasks

- [ ] M5.1 Implement fixed-region SSIM + color diagnostic, check region registration/margins/empty region.
- [ ] M5.2 Ghi CLIP full/crop/gain với domain text policy freeze; kiểm tra style bias và background sẵn chứa target.
- [ ] M5.3 Chốt implementation S_b, detector/morphology/version; nếu adapted thì đổi tên rõ.
- [ ] M5.4 Annotation actual boundary trên subset để kiểm tra expected-ROI mismatch; portrait outer boundary không thay face structure.
- [ ] M5.5 Controls: background-only, sticker, blur/noise, malformed structure, shifted ROI.
- [ ] M5.6 Phân tích metric–human rank association, pairwise agreement và failures theo domain/style.
- [ ] M5.7 Chọn diagnostic-only vs possible gate; freeze thresholds trên calibration data và đánh giá validation tách biệt.
- [ ] M5.8 Ghi false accepts, false rejects, uncertainty và sample-size limitations, không chỉ average correlation.

### Deliverables

Versioned scorers/eval assets, `metric_validation.md`, controls manifest, plots/tables và quyết định explicit cho từng metric.

### Gate

Có evidence để quyết định cách dùng metrics. **Metric không tốt vẫn có thể hoàn thành mốc** nếu kết luận diagnostic-only rõ; không buộc phải tạo scalar reward cho đúng roadmap. Muốn auto gate cần validation hỗ trợ tolerance đã định trước và không bỏ qua subgroup failures.

### Failure và xử lý

Nếu S_b thưởng ảnh mất subject, không sửa label để hợp metric: giữ presence/recognition gate hoặc bỏ S_b khỏi selector. Không train reward model lớn bằng vài chục labels để che vấn đề definition.

## 9. M6 — VLM Observer và tự động hóa có điều kiện

### Mục tiêu / input

Giảm human evaluation trong loop khi có bằng chứng đủ tin cậy. Input: rubric, golden val/control set, model adapter và metrics decisions M5.

### Tasks

- [ ] M6.1 Dùng GPT-5.4 mini snapshot đã chốt cho Observer, kiểm tra hai-pass image contract, strict output và API latency/cost; không mở thêm model selection trong tuyến chính.
- [ ] M6.2 Thiết kế observer theo hai domain, không có identity task; evidence trỏ vùng ảnh và uncertainty.
- [ ] M6.3 Chấm output blind trước references nếu thử recognition proxy; ghi pass metadata rõ.
- [ ] M6.4 Không cấp optimizer hypothesis hoặc human final-test labels cho observer.
- [ ] M6.5 Measure agreement/false accept/abstention/stability theo domain, không chỉ score chung.
- [ ] M6.6 Thêm evaluator gates/abstain policy, selection rules và cap model calls.
- [ ] M6.7 So rule+same observer với LLM+same observer; same budget/action contract.
- [ ] M6.8 Chốt fallback: human review thì ghi human-assisted; không có human thì unresolved, không auto-pass.

### Deliverables

Observer module/config, calibration report, automated/hybrid pilot runs và end-to-end cost comparison. Ghi exact model/version/call params cho reproducibility.

### Gate

Được dùng automatic mode chỉ khi validation qua gate định lượng PRD §16.5 (precision, n, coverage và severe controls), đã preregister trước outcomes và kiểm tra cả hai domains. Nếu mẫu quá ít để kết luận, giữ hybrid/abstain. M6 optional; không phải blocker cho một bài human-assisted có claim đúng.

### Failure và xử lý

Nếu observer false-pass mặt méo nhưng “nhận ra mặt người”, tách recognition khỏi structure gates. Nếu latency lớn hơn render saving, báo trade-off và cân nhắc ít calls/rule/HITL, không chỉ báo generations saved.

## 10. M7 — Experiment chính và phân tích

### Mục tiêu / input

Kiểm định hypotheses trên locked test sau khi freeze phương pháp. Input: immutable dataset/protocol/models/methods và kế hoạch annotation thực tế.

### Tasks trước chạy

- [ ] M7.1 Chốt test source groups và domain balance; chọn trajectory-annotation subset trước khi xem outcomes.
- [ ] M7.2 Freeze method config, prompt/layout policy, seed list, N=10, selector và early-stop rule của study.
- [ ] M7.3 Chốt primary endpoint, effect size đáng quan tâm, uncertainty analysis và xử lý failed runs.
- [ ] M7.4 Lập GPU/API/human annotation budget; pilot timing trên hardware thật.
- [ ] M7.5 Export preregistration nội bộ `study_plan_v1.md` để tránh đổi protocol theo kết quả.

### Tasks khi chạy và chấm

- [ ] M7.6 Chạy methods trên same pairs/seed replicates; randomize/interleave order nếu cần giảm ảnh hưởng môi trường.
- [ ] M7.7 Ghi prefix selections 1/3/5/10 và actual costs; không selection bằng future offline labels.
- [ ] M7.8 Blind final evaluation, separate first-view/reference passes; không để rater bị primed bởi variants cùng pair.
- [ ] M7.9 Chấm 400 selected output slots main seed ×3 ratings theo first-view schedules; trajectory12 pairs×3 methods×10 trials chấm reference-aware, đặt đúng tên endpoint theo PRD §19.
- [ ] M7.10 Kiểm tra missing data/errors trước phân tích, report denominators rõ.

### Tasks phân tích

- [ ] M7.11 HumanSelectedAccept@10 là primary; ObserverSuccess@N hoặc ReferenceAwareSuccess@N cho trajectory/prefix study đúng nguồn labels, selection gap và censored cost. Không gán nhãn human first-view cho data đã primed.
- [ ] M7.12 Animal/portrait breakdown và macro-average; texture/style groups có n và CI.
- [ ] M7.13 Human active time, GPU seconds, actual steps, API tokens/cost và end-to-end latency.
- [ ] M7.14 Paired/source-group-aware uncertainty; không coi seeds/trials là độc lập.
- [ ] M7.15 Failure panels và ablations ưu tiên; report cả negative results.

### Deliverables

Frozen study bundle, raw annotations, analysis scripts, result tables/figures, failure analysis, `claims_supported.md` liệt kê claim nào có và chưa có evidence.

### Gate

Kết quả reproducible, không test-tuning, phân biệt selected image với any-found-good image và human vs automatic labels. Nếu phát hiện bug ảnh hưởng kết quả, sửa version và rerun affected methods theo protocol, giữ audit trail; không xóa những kết quả bất lợi không có lý do kỹ thuật.

## 11. M8 — Đóng gói project và trình bày

### Tasks

- [ ] M8.1 README có commands thật và minimal reproduce path, data/model prerequisites rõ.
- [ ] M8.2 Bundle configs/manifests/protocol versions; outputs samples và source provenance.
- [ ] M8.3 Architecture diagrams phản ánh code đã có; optional observer/hint dùng nét/nhãn planned nếu chưa implement.
- [ ] M8.4 Slides: problem → current pipeline → target/rubric → controller → data → baselines → results → failure/limits.
- [ ] M8.5 Portrait scope ghi “face structure and integration”, không “identity preservation”.
- [ ] M8.6 Related-work comparison và contribution statement dựa trên kết quả, không chỉ số agents.
- [ ] M8.7 Kiểm tra một reproduction run từ bundle trong môi trường được mô tả.
- [ ] M8.8 Archive known issues và future work có ưu tiên, không trộn vào completed features.

### Gate

Người khác có thể hiểu inputs/outputs, chạy experiment nhỏ và truy được số trong báo cáo về run/config/labels. Không cần public release dữ liệu chưa có quyền chia sẻ; manifest/protocol và available artifacts phải ghi phạm vi rõ.

## 12. Acceptance test matrix

| Scenario | Expected behavior | Mốc |
|---|---|---|
| Prompt label trống/sai domain | Reject manifest hoặc yêu cầu sửa trước render | M0/M1 |
| NaN hoặc tham số ngoài whitelist | Reject proposal; bounded repair | M3/M4 |
| VLM/API timeout | Log call/retry/cost; fallback policy explicit | M4/M6 |
| Trial cuối rendered, đang chờ human | Resume chấm cùng ảnh, không thêm render | M3 |
| Crash trước khi lưu output hợp lệ | Mark interrupted attempt, không miễn phí retry | M3 |
| Generation mask bị đổi ngoài policy | Contract validation fail; eval assets không đổi theo | M3/M5 |
| Background-only đạt SSIM rất cao | Recognition fail, không accept | M5/M6 |
| Portrait đọc ra mặt nhưng sai cấu trúc | Face-structure fail dù face recognized | M2/M6 |
| Portrait khác người nguồn nhưng cấu trúc/blend tốt | Không reject vì identity | M2/M7 |
| Có ảnh accepted trong history nhưng selector chọn ảnh xấu | SelectedAccept fail, Success có thể pass; report gap | M7 |
| Tập nhỏ/source lặp | Báo limitation và grouped analysis, không claim universal coverage | M7/M8 |

### 12.1. Thứ tự các gói implementation có thể review

| Gói | Files/modules chính | Acceptance trước khi merge/release |
|---|---|---|
| I01 — Input integrity | Normalization collector, manifests, tests fixtures RGBA/portrait | Hidden RGB không ảnh hưởng composite; giữ tỷ lệ; hashes/provenance |
| I02 — Renderer instrumentation | `runner.PairResult`, renderer wrapper | Steps_run thật, stage timers, no generation algorithm change |
| I03 — Evaluation data | eval assets/schema/human form | Registration và blind-view tests; source/test labels separation |
| I04 — Search core | policy/history/controller/selector | Whitelist/budget/final-trial-resume/error tests |
| I05 — Baselines | random/rule/ordered configs | Same contract/selector; random state và rule IDs log |
| I06 — Text proposer | Responses adapter/schema/prompts | Fixture invalid/truncated/refusal handling + bounded live pilot |
| I07 — Metrics | fixed SSIM/CLIP diagnostics/boundary experiment | Metric version/control study; no automatic reward assumption |
| I08 — Observer | A/B image packaging + diagnosis schema | No target leakage A, aligned reference B, val gate report |
| I09 — Study/export | benchmark scheduler/annotation assignment/analysis | Rater exposure checks, exact counts, prefix label naming, portable bundle |

Các file mới là mục tiêu implementation, chưa được tạo trong lần review tài liệu. Mỗi gói có thể chia nhỏ nhưng không trộn model integration với thay core diffusion trong cùng change để khó truy nguyên kết quả.

## 13. Backlog ưu tiên để bắt đầu code

### P0 — trước model calls

1. Contracts/manifest và labels hai domains.
2. Input collection/QA và audit run collector.
3. Independent evaluation assets và annotation storage.
4. Three-key validator và controller state/resume.
5. Random/rule/human baseline + selector.

### P1 — hoàn thành MVP

6. One-LLM proposer/backend adapter và bounded repair.
7. UI Search/Annotation và portable export.
8. Pilot comparison, history ablation, cost analysis.

### P2 — chỉ khi có evidence/data

9. Metric calibration và optional VLM observer.
10. Automatic gates/abstain và locked benchmark.
11. Separate hint/retrieval/init improvements bằng ablation.

## 14. Phiếu cập nhật sau mỗi mốc

```markdown
# Milestone report: Mx

- Code/config/dataset versions:
- Tasks completed:
- Tasks pending:
- Inputs and run IDs:
- What was measured:
- Observed results, including failures:
- Functional checks:
- Research outcome: useful / unclear / no benefit / not evaluated
- Gate decision and evidence:
- Scope changes, if any:
- Next tasks:
```

Không điền số giả để hoàn thiện báo cáo. Null/pending là trạng thái hợp lệ khi chưa đo. PRD và kế hoạch hiện tại hoàn tất phần thiết kế; các tasks triển khai/data ở trên vẫn chưa được đánh dấu done.
