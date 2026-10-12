# CamouBlend — Hướng dẫn chọn ảnh mẫu và xây golden set

Ngày cập nhật: 03-10-2026.

**Cập nhật phạm vi: bạn chưa có output mẫu; subject gồm động vật và human portrait (chân dung người). Bắt đầu từ [kế hoạch chọn input](INPUT_SELECTION_PLAN.md): 6 subjects + 6 backgrounds, audit 3 pairs trước. Sau khi render mới chọn khoảng 6 output làm ảnh neo. Chưa cần xây hai agent hoặc tự động hóa reward.**

Tài liệu này là hướng dẫn thực hành bổ sung cho [PRD CamouBlend](PRD_CAMOUBLEND.md). Những số lượng, mức tham số và lịch làm việc dưới đây là đề xuất pilot; chưa phải thiết lập đã được thực nghiệm xác nhận. Chưa có ảnh của bạn được tuyển chọn hoặc chấm trong tài liệu này.

**Lưu ý chân dung đã chốt:** chỉ giữ hình dáng khuôn mặt và hòa nền đẹp, không yêu cầu danh tính người nguồn. Chấm cấu trúc mắt/mũi/miệng và integration; không có identity reward/failure. Khi xây split, group cùng người/nguồn để tránh leakage từ ảnh liên quan, không phải để thêm nhiệm vụ face recognition. Xem [PRD](PRD_CAMOUBLEND.md), [milestones](MILESTONES_CAMOUBLEND.md) và [kế hoạch input](INPUT_SELECTION_PLAN.md).

## 1. Chốt lại mục tiêu trước khi chọn ảnh

Bạn đã xác nhận: **đối tượng cần hòa vào nền và người xem nhận ra ngay**.

Một output đạt cần đồng thời:

1. Người xem nhận ra đúng loại đối tượng ở lần nhìn đầu, không cần được chỉ vị trí.
2. Đối tượng hòa vào màu sắc, texture và cấu trúc của nền; không giống một miếng sticker dán lên ảnh.
3. Hình dáng đủ hợp lý; không mất bộ phận quan trọng hoặc sinh thêm bộ phận gây lỗi rõ ràng.
4. Background ngoài vùng chèn subject vẫn giữ nội dung và bố cục cần bảo toàn.
5. Không có lỗi nổi bật như halo, seam, mảng nhòe hoặc texture hỏng.

Không đòi giữ nguyên màu lông/texture của subject nguồn nếu mục tiêu của generator là biến đổi appearance để hòa nền. Cần phân biệt **đúng loại và cấu trúc đối tượng** với **giữ nguyên identity/appearance ảnh nguồn**; pipeline hiện tại chưa bảo đảm điều thứ hai.

Một ảnh chỉ thấy đối tượng sau khi được nói “có con mèo ở đây” chưa chứng minh được yêu cầu nhận ra ngay. Tác giả đã biết subject nên không thể tự cung cấp phép đo nhận dạng không gợi ý; nhận xét của tác giả vẫn hữu ích để chốt thẩm mỹ ban đầu.

## 2. Phân biệt bốn loại dữ liệu

| Bộ dữ liệu | Câu hỏi nó trả lời | Cách chọn | Được dùng vào việc gì? |
|---|---|---|---|
| Reference examples — ảnh mẫu định nghĩa mục tiêu | Bạn gọi ảnh nào là đẹp/đạt? | Chọn có chủ đích 4–6 outputs | Thảo luận với cô, viết rubric, minh họa lỗi |
| Audit set — bộ kiểm tra ban đầu | Tham số tác động ra sao; generator có tạo được ảnh đạt không? | Chọn 3 dev pairs, mỗi pair 7 cấu hình cố định | Kiểm tra preprocessing, sensitivity và logging |
| Golden dev/validation | Feedback và metric có đáng tin không? | Chọn inputs có độ phủ; giữ cả success/failure | Thiết kế action space, calibration, few-shot dev-only |
| Locked test | Phương pháp có tốt trên dữ liệu chưa dùng để thiết kế không? | Chốt nguồn ảnh/split trước khi xem kết quả của phương pháp | Đánh giá cuối, không dùng sửa prompt/rule/threshold |

**Golden set không có nghĩa “chỉ giữ ảnh đẹp”.** Nó bao gồm ảnh đạt, ảnh lỗi, ảnh khó quyết định, config và raw judgments.

Các ảnh bạn đã xem kỹ hoặc dùng để sửa phương pháp nên đưa vào reference/dev. Không đưa lại chúng vào test rồi coi là dữ liệu chưa thấy.

## 3. Sau khi render audit, chọn 6 output mẫu như thế nào?

### 3.1. Danh sách chọn đề xuất

| ID mẫu | Loại nên tìm | Cụ thể nên thấy gì? | Vì sao cần? |
|---|---|---|---|
| R01 | Đạt rõ ràng | Nhận ra đối tượng; hòa nền tự nhiên; không lỗi lớn | Neo cho định nghĩa “accept” |
| R02 | Đạt nhưng khác R01 | Subject hoặc nền khác; vẫn đạt theo cùng tiêu chí | Tránh định nghĩa chất lượng chỉ từ một ảnh đẹp |
| R03 | Quá nổi/sticker | Nhận ra ngay nhưng outline, màu hoặc texture tách khỏi nền | Tách recognition khỏi integration |
| R04 | Quá chìm/mất subject | Có thể nền đẹp nhưng khó nhận ra hoặc subject không còn | Kiểm tra reward không thưởng việc xóa subject |
| R05 | Lỗi cấu trúc hoặc background | Sai chi, mặt biến dạng, hoặc vùng nền bị thay đáng kể | Phân biệt blend chưa đẹp với lỗi nghiêm trọng |
| R06 | Trường hợp phân vân | Bạn thấy “gần đạt” hoặc khó thống nhất | Xây vùng uncertain, tránh ép mọi ảnh thành 0/1 |

Nếu có đủ dữ liệu, thêm riêng một ảnh `background_damaged` và một ảnh `artifact` để chúng không bị gộp vào R05.

Không bắt buộc mỗi ảnh chỉ có một lỗi. Ví dụ R03 có thể vừa sticker vừa sai cấu trúc; ghi tất cả lỗi và chỉ rõ lỗi chính.

### 3.2. Ưu tiên những nhóm ảnh có thể so trực tiếp

Giá trị nhất là tìm được **cùng một cặp input**, nhưng có:

```text
Output A: quá nổi
Output B: đạt
Output C: quá chìm
```

Nếu config/seed được lưu, nhóm này giúp hiểu sự đánh đổi. Nếu seed khác hoặc nhiều tham số cùng đổi, vẫn dùng làm ví dụ thẩm mỹ nhưng chưa kết luận tham số nào gây cải thiện.

Không cần đủ ba trạng thái mới bắt đầu. Với ảnh sẵn có, chọn gần nhất với danh sách và ghi phần còn thiếu.

### 3.3. Nếu chưa có ảnh “đạt rõ ràng”

Giữ ảnh tốt nhất hiện có nhưng gắn `best_available_not_accepted`. Viết cụ thể còn thiếu điều gì. Không đổi nhãn thành accept chỉ để đủ bộ.

Bạn có thể thêm ảnh tham khảo bên ngoài để giải thích thẩm mỹ mong muốn, nhưng ghi `external_reference`. Không gọi đó là output CamouBlend hoặc đưa vào benchmark hiệu năng generator. Ghi nguồn ảnh và điều kiện sử dụng nếu sau này đưa lên slide/paper.

Nếu chưa có output nào, chuyển thẳng sang audit 3 pairs × 7 configs ở mục 7; sau đó chọn reference examples từ kết quả dev.

### 3.4. Không nên chọn thế nào?

- Không chỉ chọn ảnh được (S_b), CLIP hoặc SSIM chấm cao.
- Không chỉ chọn cùng một đối tượng trên cùng loại nền.
- Không lấy screenshot bị thu nhỏ nếu còn file gốc.
- Không chỉnh sáng, sharpen hoặc crop riêng một output rồi so với output chưa chỉnh; bản minh họa chỉnh sửa phải tách riêng.
- Không bỏ ảnh xấu khỏi log sau khi đã render.
- Không coi việc người xem nhận ra sau khi được cho biết nhãn đối tượng là recognition không gợi ý.

## 4. Mỗi ảnh mẫu cần kèm những gì?

### 4.1. Gói tối thiểu để mình và cô xem được

Với mỗi output, chuẩn bị:

1. Ảnh subject gốc.
2. Ảnh background gốc.
3. Output nguyên bản.
4. Nhận xét của bạn: đạt/chưa đạt/phân vân và lý do.
5. Config hoặc những tham số/seed còn nhớ; phần không biết ghi `unknown`.

Nếu một subject–background có nhiều output, dùng chung input pair ID; mỗi output có sample ID riêng.

### 4.2. Gói nên có để tái lập

- Full config đã resolve, không chỉ ba tham số đang tối ưu.
- Prompt subject thực tế, background prompt và negative prompt.
- Seed, seed mode, số steps đặt và số steps thực chạy.
- Layout: scale, offset, rotation; resolution và quy tắc resize/crop.
- Checkpoint/model revision, scheduler, code revision và môi trường.
- Canny, SoftEdge/HED và generation mask ở đúng tọa độ generator.
- Runtime; lỗi/warnings nếu có.

Config cũ bị mất không làm ảnh vô dụng: vẫn dùng nó để xây rubric, nhưng gắn `reproducibility: incomplete` và không dùng nó làm bằng chứng hiệu quả tuning.

### 4.3. Mẫu ghi chú có thể điền ngay

```markdown
# R01

- Pair ID:
- Subject label:
- Subject file:
- Background file:
- Output file:
- Config file: unknown nếu không còn
- Seed: unknown nếu không còn
- Vai trò ảnh: accepted / too_visible / missing / structural_error / uncertain
- Quyết định của tác giả: accept / reject / uncertain
- Tôi nhận ra đối tượng vì:
- Vùng nào hòa nền tốt:
- Vùng nào còn lỗi:
- Background thay đổi ngoài ý muốn ở đâu:
- Một thay đổi tôi muốn ở output tiếp theo:
- Người nhận xét đã biết subject trước khi xem chưa: yes / no
```

Ví dụ nhận xét hữu ích: “Đầu và thân đọc được nhưng viền lưng quá sáng so với đá phía sau”. Nhận xét “ảnh chưa đẹp” chưa đủ để đề xuất action có căn cứ.

## 5. Chọn subject và background cho 3 cặp audit

### 5.1. Chọn phạm vi style trước

Nếu phần lớn dữ liệu hiện tại là silhouette/line-art, lấy style đó làm pilot chính. Nếu là ảnh động vật tự nhiên, lấy ảnh tự nhiên làm pilot chính. Chưa nên trộn nhiều style chỉ để trông đa dạng khi mỗi style chỉ có một ví dụ.

Tất cả inputs ở đây là ảnh raster 2D. Ảnh chụp động vật có cảm giác 3D không tự tạo ra depth/mesh; không cần tìm 3D ground truth.

### 5.2. Subject nên có đặc điểm gì?

Ưu tiên input phù hợp với use case thực của bạn:

- Một subject chính: animal có nhãn nhóm rõ; portrait có mặt rõ và góc nhìn được ghi.
- Các bộ phận quan trọng đủ trong khung; pose dễ hiểu ở resolution sử dụng.
- Silhouette/đường nét đủ rõ để kiểm tra edge extraction.
- Chất lượng ảnh đủ dùng, không phụ thuộc việc upscale mạnh.
- Với pilot đầu, tránh nền subject quá rối khiến HED lấy cả texture nền làm cấu trúc đối tượng.

Đừng chỉ giữ input “sạch hoàn hảo” mãi: khi audit chạy được, thêm cases nền subject phức tạp vào dev và gắn tag riêng. Nếu chúng nằm trong phạm vi sản phẩm, failure preprocessing cũng là kết quả cần báo.

Ví dụ hình dáng để tăng độ phủ: thân khối rõ, thân dài/mảnh, có cấu trúc nhỏ như đuôi/chân/cánh. Đây là nhóm hình thái minh họa, không bắt buộc dùng loài cụ thể.

### 5.3. Background nên có đặc điểm gì?

- Có vùng đặt subject đủ diện tích, nội dung phù hợp nhiệm vụ.
- Không có sẵn đối tượng mục tiêu trong pilot đầu; nếu có, ghi tag vì có thể làm CLIP hoặc người xem nhận sai nguồn subject.
- Không có chữ/watermark lớn gây chi phối đánh giá.
- Phủ mức texture thấp, vừa và cao.
- Giữ scene/style đúng phạm vi bạn muốn nghiên cứu, không lấy toàn những cặp hoàn toàn không tương thích chỉ để tạo failure.

“Tương thích màu” xét appearance bạn muốn output hòa vào, không đòi output giữ nguyên màu subject nguồn.

### 5.4. Đặc điểm audit cần phủ

| Pair | Subject | Background | Câu hỏi kiểm tra |
|---|---|---|---|
| A01 | Hình dáng rõ, ít chi tiết mảnh | Texture vừa, vùng đặt phù hợp | Pipeline có tạo ảnh đạt trong tình huống tương đối thuận lợi không? |
| A02 | Có bộ phận mảnh/chi tiết cần giữ | Texture vừa hoặc dày | SoftEdge có giúp structure hay làm viền quá nổi? |
| A03 | Subject rõ, pose hợp lệ | Nền ít texture hoặc khác độ sáng/màu | Hòa nền và giữ background đánh đổi thế nào? |

Bảng trên mô tả đặc điểm cần phủ. Danh sách pair cụ thể hiện được chốt ở [kế hoạch input](INPUT_SELECTION_PLAN.md): mèo–cát, chân dung chính diện–tường và chân dung góc 3/4–lá dày. Đây là lựa chọn có chủ đích cho diagnosis, không phải mẫu đại diện để tuyên bố hiệu quả chung. Cả ba thuộc dev. Nên có ít nhất hai subject khác nhau nếu dữ liệu cho phép; một pair “khó” không đại diện cho toàn bộ nhóm khó.

Khóa layout cho từng pair sau khi QA. Không vừa thay position/scale, vừa thay tham số search rồi diễn giải ảnh tốt hơn là do tham số đó.

## 6. QA input và phần xử lý ảnh số trước khi render hàng loạt

**Preprocessing v1 bắt buộc:** lưu original alpha nếu có; composite rõ trên RGB(128,128,128); normalize crop/pad giữ tỷ lệ. Với ảnh opaque nền rối, chỉ đặt nền xám phía sau không tách được subject. Sau QA nếu vẫn rò, ưu tiên alpha/cutout support; learned segmentation dự phòng là BiRefNet offline, versioned và dùng chung mọi methods. Xem [PRD §6.1](PRD_CAMOUBLEND.md#61-quy-trình-bắt-buộc-cho-alpha-và-mask-leakage).

Với từng pair:

1. Xem input sau resize/crop tại resolution thực của generator.
2. Xem Canny và SoftEdge: các bộ phận subject có còn đúng không; cạnh nền có chiếm ưu thế không?
3. Xem generation mask trên background: vị trí/kích thước có khớp layout không?
4. Xem baseline output và xác định lỗi input/preprocess riêng với lỗi tuning.
5. Tạo evaluation ROI độc lập, cùng hệ tọa độ và được review.

Phân biệt các evaluation assets:

| Asset | Dùng cho gì? | Lưu ý |
|---|---|---|
| Fixed expected subject ROI | Crop để chẩn đoán và xác định vùng chèn dự kiến | Không phải silhouette thật của mọi output |
| Fixed background region | So output với background ngoài vùng chèn và margin | Không co vùng chấm lại sau khi thấy nền hỏng |
| Output silhouette annotation, calibration subset | Kiểm định boundary score theo contour subject thực | Nếu subject mất thì ghi absent; không vẽ giả mask |

Không tự chuẩn hóa màu giữa output và background trước SSIM/color diagnostics vì có thể che đi lỗi cần đo. Lưu ảnh gốc; overlay/crop là bản phục vụ kiểm tra, không thay artifact đánh giá chính.

Nếu sửa preprocessing, tăng version và áp dụng đồng nhất cho các method. Giữ lại ghi nhận lỗi cũ; không trộn outputs hai phiên bản mà không phân biệt.

## 7. Bảng 7 cấu hình cho mỗi pair audit

Default dưới đây được đối chiếu với `camo/config.py` khi soạn tài liệu. Nếu preset bạn đang dùng khác, lưu rõ actual baseline và thiết kế perturbation quanh baseline đó; không âm thầm thay số trong log.

| Config ID | sampling.bg_strength | blend.alpha_end | control.softedge_scale | So với C00 |
|---|---:|---:|---:|---|
| C00 | 0.52 | 0.54 | 1.50 | Default |
| C01 | 0.47 | 0.54 | 1.50 | Giảm bg_strength |
| C02 | 0.57 | 0.54 | 1.50 | Tăng bg_strength |
| C03 | 0.52 | 0.48 | 1.50 | Giảm alpha_end |
| C04 | 0.52 | 0.60 | 1.50 | Tăng alpha_end |
| C05 | 0.52 | 0.54 | 1.30 | Giảm SoftEdge |
| C06 | 0.52 | 0.54 | 1.60 | Tăng SoftEdge |

Bước SoftEdge tăng/giảm không đối xứng vì dùng các mức grid sơ bộ đã đề xuất; không diễn giải đây là ước lượng đạo hàm đối xứng. Audit này xem từng tham số quanh baseline, chưa khảo sát đầy đủ interactions.

Khóa những phần còn lại:

- Cùng input, prompt đã resolve và layout trong một pair.
- Cùng seed và seed mode fixed cho bảy cấu hình; default repo hiện là 1234.
- Cùng total sampling steps, checkpoint, scheduler, guidance, mask và các cấu hình còn lại.
- Cùng preprocessing assets/seed nếu preprocessing có randomness; nên cache hoặc lưu hash để kiểm tra.

`bg_strength` vẫn có thể thay số steps thực chạy mặc dù total steps config được khóa. Ghi cả hai. Cùng seed giảm một nguồn nhiễu, không chứng minh noise trajectory giống hệt khi sampling schedule thay đổi.

Không giả định tăng SoftEdge luôn cải thiện cấu trúc hoặc giảm alpha luôn làm blend đẹp hơn. Bảng trên là can thiệp để quan sát, chưa là rule được kiểm chứng.

3 pairs × 7 configs = **21 renders**. Ở 5 giây/render, phần render ước tính 105 giây. Chưa gồm load models, preprocessing, metric, annotation và biến thiên runtime giữa cấu hình.

Xem cả 7 ảnh của một pair trong contact sheet để diagnosis, nhưng dùng từng ảnh riêng ở lượt chấm recognition không gợi ý. Không dùng contact sheet có nhãn đối tượng cho phép đo recognition chính.

## 8. Cách chấm ảnh để tạo labels hữu ích

### 8.1. Hai lượt xem có mục đích khác nhau

**Lượt A — nhận dạng không gợi ý:** chỉ output. Không đưa tên file có nhãn đối tượng, prompt, ROI hoặc ảnh subject. Hỏi “Bạn thấy đối tượng gì?”. Lưu câu trả lời nguyên văn và thời gian nếu có công cụ đo.

Nếu muốn dùng time window, pilot một cửa sổ ngắn cố định, ví dụ 3 giây, rồi kiểm tra tính khả thi và chốt trước test. Ba giây là lựa chọn protocol đề xuất, không phải ngưỡng khoa học mặc định. Nếu chỉ xem thủ công không đo được thời gian, không báo số liệu reaction time chính xác.

**Lượt B — chất lượng và diagnosis:** sau khi trả lời lượt A, hiện background/subject reference. Chấm hòa nền, cấu trúc, artifacts và bảo toàn nền.

Người đã xem nhiều variants của cùng pair bị gợi ý ở các lượt sau. Với locked test, phân phối variants qua người chấm hoặc nhóm khảo sát khác; không yêu cầu một người nhận dạng “không gợi ý” cả 7 ảnh cùng đối tượng.

Tác giả có thể làm lượt B để xây rubric trước. Dữ liệu tác giả không thay thế lượt A của người chưa biết subject.

**Scheduling cuối:** primary test có 400 selected slots ×3 ratings, cần 12 counterbalanced rater slots/pair và có thể nhiều người hơn khi sources lặp. Nếu không có đủ raters, phải giảm scope trước study. Trajectory12-pair study là reference-aware; không báo Success@N của study đó như first-view recognition. Xem PRD §19 cho số lượng/cách phân công.

### 8.2. Rating anchors đề xuất

| Tiêu chí | 1 | 3 | 5 |
|---|---|---|---|
| Integration | Tách nền rõ, giống dán ảnh | Hòa một phần nhưng còn vùng lộ | Màu/texture/biên hòa tự nhiên |
| Structure | Sai/mất cấu trúc lớn | Đọc được đối tượng nhưng có lỗi | Hình dáng hợp lý trong style đã chọn |
| Background preservation | Nội dung/bố cục nền ngoài ROI bị đổi lớn | Có biến đổi đáng chú ý | Nền ngoài vùng chèn được giữ tốt |
| Overall quality | Lỗi gây khó chấp nhận | Có thể dùng sau chỉnh sửa | Có thể chọn làm kết quả cuối |

Điểm 2/4 nằm giữa anchors; cho phép `not_assessable`. Ratings là thứ bậc, chưa mặc định chênh 1→2 tương đương chênh 4→5, và chưa cộng trung bình thành reward.

Recognition ghi riêng `correct / incorrect / uncertain`, cùng câu trả lời gốc. Chốt trước granularity nhãn và synonyms: ví dụ cần nhận ra “mèo”, hay phân biệt một giống mèo cụ thể? Pilot nên theo mức nhãn mà inputs và mục tiêu thực sự hỗ trợ.

### 8.3. Accept/reject/uncertain

- **Accept:** đúng subject ở mức nhãn đã định, hòa nền đủ tốt, không lỗi nghiêm trọng và giữ nền theo yêu cầu.
- **Reject:** có ít nhất một điều kiện thiết yếu thất bại; ghi lý do.
- **Uncertain:** người chấm chưa quyết định được hoặc thiếu bằng chứng; không ép thành accept.

Chưa áp dụng máy móc “mọi rating ≥4” trước khi đối chiếu các ảnh neo. Khi có anchors, chốt quy tắc tổng hợp labels trước locked test. Giữ raw votes kể cả khi tạo consensus sau này.

### 8.4. Failure taxonomy

| Tag | Diễn giải | Tránh nhầm với |
|---|---|---|
| subject_missing | Subject không tồn tại hoặc gần như không đọc được | Đã hòa nền đẹp |
| subject_too_hidden | Có dấu hiệu subject nhưng không nhận ra ở lượt đầu | Thành công camouflage theo mục tiêu hiện tại |
| subject_too_visible | Subject nổi tách nền, sticker/outline rõ | Chỉ đơn giản là nhận ra ngay |
| wrong_category (animal only) | Nhận ra sai nhóm động vật yêu cầu | Thay màu/texture hợp lệ |
| face_structure_error (portrait) | Mắt/mũi/miệng hoặc outline mặt méo, sai cấu trúc | Trông khác danh tính người nguồn |
| wrong_structure | Sai chi, đầu, thân hoặc cấu trúc quan trọng | Khác appearance nguồn |
| background_damaged | Nền ngoài vùng chèn thay đổi không mong muốn | Thay đổi cần thiết bên trong subject |
| artifact | Halo, seam, nhòe hoặc pattern lỗi | Texture nền tự nhiên |

Các tags là multi-label. `acceptable` là quyết định riêng, không phải một failure tag. `uncertain` là trạng thái đánh giá; có thể đi kèm ghi chú về tag chưa chắc chắn.

### 8.5. Pairwise preference

Sau khi có ratings, có thể so candidate với default hoặc incumbent:

```text
Bạn chọn ảnh A, ảnh B, hòa, hay chưa quyết định được?
Cả hai có thể vẫn chưa đạt.
Lý do: recognition / integration / structure / background / artifacts.
```

Đảo ngẫu nhiên vị trí A/B, giữ ties và uncertainty. Không mặc định ảnh thắng pairwise là ảnh đạt.

## 9. Từ 3 audit pairs mở rộng thành golden pilot

### 9.1. Quy mô thống nhất v1.1

| Stage | Pairs | Domain balance | Dùng để |
|---|---:|---|---|
| Pilot | 18, nằm trong dev | 12 animal / 6 portrait | Input/preprocessing/sensitivity audit |
| Dev | 120 tổng, gồm pilot | 60 / 60 | Rubric, few-shot, rule/prompt, calibration |
| Validation | 30 nguồn độc lập | 15 / 15 | Frozen observer/evaluator check |
| Locked test | 100 nguồn độc lập | 50 / 50 | Primary evaluation |

Đây là planning targets, không phải data đã thu hoặc power guarantee. Không dùng split 8/4/6 nữa. 18 pairs dùng chung sáu subjects/sáu backgrounds đều là dev. Nếu chưa đủ nguồn/raters, báo pilot scope; không gọi nó là full benchmark.

Quotas representation×local texture và output counts nằm ở PRD §15; cả bốn tài liệu dùng cùng kế hoạch này. Giữ source/person/background groups trong một split; ảnh khác của cùng nguồn không tạo independence.

### 9.2. Chọn pairs trước khi thấy outputs của phương pháp

Tạo inventory inputs rồi gắn tags:

- Subject style: silhouette / line-art / natural-photo / other.
- Shape: compact / elongated / fine-parts.
- Background texture: low / medium / high.
- Appearance compatibility dự kiến: favorable / moderate / challenging / unknown.
- Source group và near-duplicate group.
- Nguồn ảnh, quyền sử dụng và lý do loại nếu có.

Chia groups subject và background trước khi ghép pair nếu muốn test cả hai đều chưa thấy. Giữ bản crop/resize của cùng ảnh nguồn trong cùng group. Nếu chỉ pair-disjoint, báo rõ test tổ hợp mới chứ không claim cả subject/background đều mới.

Chọn coverage phù hợp quy mô; pilot 18 pairs không thể cân bằng mọi tổ hợp style × shape × texture; dùng quotas PRD §15 khi mở rộng. Ưu tiên style chính của dự án, ghi các nhóm chưa có dữ liệu.

Nếu pair thất bại sau render nhưng input vẫn hợp lệ, giữ nó trong analysis. Chỉ loại theo tiêu chí kỹ thuật đã chốt và ghi lý do; không loại vì agent không thắng.

### 9.3. Dự toán outputs

- Dev: 120 default outputs + 12 audit pairs×6 additional configs = tối đa 192 outputs. Ba audit pairs đầu nằm trong 12 pairs, không cộng lần nữa.
- Manual dev: 6 pairs (3/domain)×tối đa 4 renders mới ngoài default = thêm tối đa 24, tổng 216 trước dedup.
- Validation: 30 pairs×C00/C03/C06 = 90 natural outputs, thêm 24 controls phân tích riêng.
- Test: theo core protocol PRD §19; không sweep test rồi chọn những pairs đẹp.

Golden labels reference-aware phù hợp chẩn đoán/calibration. First-view final human study phải dùng scheduling riêng để rater không biết trước target qua variants/source lặp. Không gọi toàn bộ golden labels là unbiased recognition.

Nếu sửa evaluator sau validation, ghi adaptation/version và dùng validation mới cho gate độc lập. Locked test không được dùng tune.

## 10. Thu manual trajectory như thế nào?

Mỗi bước lưu:

```text
Ảnh hiện tại → nhận xét trước khi chỉnh → action đã chọn
→ config mới → output → nhận xét sau khi xem → quyết định tiếp tục/dừng
```

Ghi cả parent trial nếu quay lại cấu hình cũ. Lưu action hypothesis trước render để phân biệt dự đoán với giải thích sau khi nhìn kết quả.

Đo riêng:

- Render/preprocessing/metric latency.
- Thời gian người xem và quyết định action.
- Thời gian chờ/gián đoạn không làm việc nếu có thể phân biệt.

Không dựng lại một trajectory “luôn tốt dần” bằng cách sắp các output cũ. Nếu chỉ có outputs lịch sử không timer, dùng chúng cho qualitative analysis, không claim tiết kiệm thời gian human tuning từ chúng.

Human baseline chính dùng cùng ba tham số và bounds của optimizer. Nếu con người được đổi prompt, mask hoặc layout, tách thành một setting khác.

## 11. Metric nào tính ngay, metric nào chưa dùng làm reward?

### 11.1. Ưu tiên hiện tại

| Signal | Ghi ngay? | Dùng tự động chọn ảnh ngay? |
|---|---|---|
| Human accept/failure/preference | Có | Theo selector human-assisted đã định |
| Human recognition không gợi ý | Có khi có người chưa biết subject | Là endpoint đánh giá; test labels không đưa ngược vào search |
| SSIM nền trên fixed region | Có sau QA vùng đánh giá | Chưa, cần calibration |
| CLIP semantic similarity | Có như diagnostic | Chưa, không tự thay recognition |
| S_b | Sau khi chốt implementation/assets | Chưa |
| S_edge/hidden_score hiện có | Có thể lưu | Không coi tương đương S_b hay human acceptance |

### 11.2. Tại sao không chọn ảnh theo S_b cao nhất?

Định nghĩa gốc:

\[
S_b(I,m)=1-F_1(m_b\odot C_{gt},m_b\odot C),
\qquad m_b=\mathrm{dilate}(m)-\mathrm{erode}(m).
\]

C là contour ảnh, C_gt là contour foreground mask. Điểm cao thể hiện biên subject ít rõ hơn; nó không bảo đảm subject còn tồn tại hoặc được nhận ra. Định nghĩa đã được đối chiếu với [paper gốc ICCV 2023](https://arxiv.org/html/2309.03899v1); chưa kiểm định thực nghiệm trên CamouBlend. Repo tham khảo: [CAMEVAL](https://github.com/hlamdouar/CAMEVAL).

Controls cần giữ trong tập metric validation:

1. Background-only: phải thất bại điều kiện presence/recognition dù nền được giữ hoàn hảo.
2. Subject rõ như sticker: có thể nhận ra đúng nhưng không đạt integration.
3. Blur/noise: metric không được dẫn selector đến ảnh hỏng chỉ vì contour khó thấy.
4. Shape/ROI mismatch: kiểm tra score có đẹp giả vì đo nhầm biên không.

Controls là bài kiểm tra có chủ đích, không trộn chúng vào acceptance rate của natural benchmark như các outputs thông thường.

Mọi detector, threshold, morphology, crop và empty-contour handling phải có version. Nếu thay định nghĩa gốc bằng Canny/HED hoặc fixed expected ROI, ghi tên metric điều chỉnh, không gọi là tái hiện chính xác S_b gốc.

Chỉ đưa metric vào loop khi đã đối chiếu human labels, xem false acceptance và kiểm tra trên validation. Nếu không align, giữ làm diagnostic và tiếp tục human feedback.

## 12. Cấu trúc lưu dữ liệu đề xuất

Cây dưới là đề xuất, chưa có các ảnh hoặc thư mục dữ liệu này được tạo:

```text
benchmarks/golden_v1/
  README.md
  pairs.jsonl
  reference_examples.csv
  protocols/
    rubric_v1.md
    action_space_v1.json
    split_v1.json
  assets/
    subjects/s001.png
    backgrounds/b001.png
  evaluation_assets/p001/
    expected_subject_roi.png
    background_region.png
    metadata.json
  reference_examples/R01/
    notes.md
  runs/run_p001_audit/
    run.json
    C00/
      config.yaml
      output.png
      metrics.json
      cost.json
      debug/
    C01/
      ...
  annotations.jsonl
  comparisons.jsonl
```

Không lưu API keys hoặc token truy cập trong configs/logs. Annotation của locked test để ngoài context mà proposer/retriever đọc được.

### 12.1. Bảng inventory tối thiểu

```csv
pair_id,split,subject_id,background_id,subject_label,subject_style,background_texture,subject_group,background_group,status
p001,dev,s001,b001,lizard,silhouette,medium,sg001,bg001,pending
```

Dòng này chỉ minh họa schema; không phải dữ liệu đã có. Full manifest bổ sung paths/hashes, nguồn, prompt/layout và evaluation assets version.

### 12.2. Annotation record minh họa

```json
{
  "annotation_id": "a001",
  "output_id": "run_p001_audit_C00",
  "rater_id": "rater_01",
  "rubric_version": "v1",
  "reference_seen_before_first_view": false,
  "recognition": {
    "raw_answer": null,
    "mapped_result": null,
    "response_time_ms": null
  },
  "acceptance": null,
  "failure_tags": [],
  "ratings": {
    "integration": null,
    "structure": null,
    "background_preservation": null,
    "overall_quality": null
  },
  "notes": null
}
```

`null` là chưa đo/chưa chấm, không phải 0 hoặc reject. Không đưa score tự khai của LLM vào trường runtime hoặc human label.

## 13. Khi nào đủ để chuyển sang xây optimizer?

Sau audit, lập một bảng cho từng pair:

| Câu hỏi | Nếu câu trả lời là “chưa” |
|---|---|
| Inputs/edge/mask có hợp lệ không? | Sửa input/preprocessing và ghi version trước |
| Có output đạt hoặc gần đạt với điều kiện lỗi rõ không? | Kiểm tra năng lực generator và miền tham số trên dev |
| Người chấm hiểu accept/reject/uncertain giống nhau ở mức đủ dùng không? | Bổ sung ảnh neo, sửa rubric và giữ disagreements |
| Có log đủ để tái lập và so tham số không? | Sửa collector trước khi tăng số ảnh |
| Biết ít nhất vài action có ích trong một số tình huống không? | Thu thêm sensitivity/interaction evidence |

Không cần mọi pair đều thành công mới xây optimizer. Cần biết generator có phạm vi thành công và failure nào có thể do search; nếu không pair nào đạt thì agent chưa có cơ sở để tối ưu.

Thứ tự triển khai sau đó:

1. Random và rule/coordinate baseline trên cùng action space.
2. Một text LLM Config Proposer đọc history + human feedback.
3. Đo chất lượng, render cost và human active time.
4. Thêm VLM Observer khi có labels để calibration.
5. Chỉ thêm hint call/retrieval khi ablation cho lợi ích.

Pilot controller dùng 5 renders; benchmark chính có thể dùng 10 và báo prefix 1/3/5/10. Đó là budget search, tách với 7-config audit để nghiên cứu sensitivity.

## 14. Lịch thực hiện theo buổi

Đây là thứ tự deliverables, không phải cam kết mỗi phần chỉ mất một buổi.

| Buổi | Việc cần hoàn thành | Sản phẩm |
|---|---|---|
| 1 | Thu subject/background theo kế hoạch input, ghi source/style/tags | Input inventory; chưa cần có outputs |
| 2 | Chọn 3 dev pairs, QA edges/mask/layout, khóa baseline | Manifest nhỏ + debug assets |
| 3 | Chạy 21 renders và lưu mọi output/config/cost | Audit set có thể tái lập |
| 4 | Chấm outputs mới; chọn khoảng 6 ảnh neo; tổ chức first-view nếu có người mới | Reference examples, labels, disagreements và rubric v1 |
| 5 | Tổng hợp sensitivity và xác định gaps | Action-space đề xuất, quyết định mở rộng dev |
| Sau đó | Inventory/split golden pilot, thu trajectories, chạy baselines | Dữ liệu đủ để xây và kiểm chứng proposer |

## 15. Checklist chuẩn bị gửi để được review

### Gói outputs — chuẩn bị sau audit

Trước gói này, chuẩn bị inputs theo [INPUT_SELECTION_PLAN.md](INPUT_SELECTION_PLAN.md); không yêu cầu có outputs sẵn.

- [ ] Khoảng 6 output theo R01–R06; không ép đủ ảnh đạt nếu chưa có.
- [ ] Subject và background tương ứng với từng output.
- [ ] Nhận xét ngắn giải thích đạt/lỗi/phân vân.
- [ ] Config và seed nếu còn; phần thiếu ghi unknown.
- [ ] Xác định style chính đang làm: silhouette, line-art hay natural photo.
- [ ] Ảnh gốc đủ resolution, không chỉ screenshot.

### Gói audit tiếp theo

- [ ] 3 dev pairs có ID và label chính xác.
- [ ] Prompt/layout/preprocessing và seed được khóa theo pair.
- [ ] 7 config IDs C00–C06 cho mỗi pair.
- [ ] Output, full config, cost và debug assets cho mọi trial.
- [ ] Annotation giữ cả failures và uncertainty.
- [ ] Evaluation assets độc lập generation mask.

### Trước khi báo cáo kết quả nghiên cứu

- [ ] Split không rò nguồn ảnh hoặc công bố rõ giới hạn split.
- [ ] Rubric/threshold/prompt/selector được freeze trước locked test.
- [ ] Cùng generator, bounds, seed policy và budget giữa các methods.
- [ ] Người chấm cuối độc lập, không biết method/config.
- [ ] Báo ảnh hệ thống thực sự chọn, không chỉ ảnh tốt nhất tìm được sau khi xem toàn bộ.
- [ ] Báo cả runs thất bại, compute, API và thời gian người.

**Sản phẩm cần có ngay lúc này là bộ ảnh mẫu có giải thích và một audit nhỏ có log. Chúng quyết định evaluator và action space; kiến trúc agent được xây sau để phục vụ tiêu chí đã rõ.**
