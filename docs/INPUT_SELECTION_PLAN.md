# CamouBlend — Chọn input từ đầu khi chưa có ảnh mẫu

Cập nhật: 03-10-2026. Đây là kế hoạch thu thập và thử nghiệm đề xuất, chưa phải dữ liệu hoặc kết quả đã có.

**Tài liệu chuẩn:** [PRD CamouBlend](PRD_CAMOUBLEND.md) và [kế hoạch milestone](MILESTONES_CAMOUBLEND.md). Người dùng đã xác nhận portrait chỉ cần giữ hình dáng mặt và hòa nền đẹp, không yêu cầu giữ danh tính.

## 1. Quyết định cho giai đoạn hiện tại

Bạn chưa có bộ output mẫu. Subject gồm **động vật và human portrait (chân dung người)**. Lá, chìa khóa, cốc và các đồ vật khác nằm ngoài phạm vi hiện tại. Chân dung không tự mở rộng sang toàn thân người. Background có thể là ảnh chụp, minh họa/pattern 2D hoặc ảnh render 3D, từ nền đơn điệu đến nhiều chi tiết.

**Bắt đầu bằng 6 subject images và 6 background images, ghép có chọn lọc thành 18 dev pairs. Audit 3 pairs trước, sau đó mới mở rộng. Không cần có sẵn ảnh đẹp để làm bước này.**

Trình tự:

```text
Thu input → QA input/edges/layout → render audit
→ chấm outputs → chọn ảnh neo cho rubric
→ hoàn thiện golden dev → baselines → optimizer/agent
```

“Golden” ở đây là dữ liệu có metadata và đánh giá đáng tin, không phải bộ toàn ảnh đẹp. Toàn bộ 18 cặp khởi đầu dưới đây là development data; chưa phải benchmark test độc lập.

Mục tiêu vẫn là: **đối tượng hòa vào background nhưng nhận ra ngay**, giữ nền ngoài vùng chèn và không có lỗi hình dáng nghiêm trọng. Nền phẳng có thể ít cơ hội hòa texture hơn nền tự nhiên; không đổi mục tiêu thành “khó tìm ra” chỉ vì một loại nền khó xử lý.

## 2. Phân loại background bằng những trục độc lập

Không nên đặt “2D”, “3D”, “thiên nhiên”, “phức tạp” vào một danh sách loại trừ nhau. Một ảnh render 3D có thể chỉ có một bức tường phẳng; một pattern 2D có thể cực kỳ nhiều chi tiết.

| Trục | Giá trị đề xuất | Ý nghĩa |
|---|---|---|
| Kiểu biểu diễn | photograph, illustration_2d, procedural_pattern_2d, rendered_3d, unknown | Ảnh được chụp/vẽ/tạo theo cách nào |
| Nội dung/vật liệu | wall, sand, foliage, wood, stone, geometric_pattern, ... | Nền chứa gì |
| Mật độ chi tiết cục bộ | low, medium, high | Chi tiết tại vùng sẽ chèn object, không chỉ toàn ảnh |
| Tính lặp | none, irregular, regular | Texture ngẫu nhiên hay có quy luật lặp |
| Hướng texture | isotropic, directional, mixed | Không hướng rõ, một hướng trội hay nhiều hướng |
| Độ nổi khối trong ảnh | flat, shallow, strong | Ánh sáng/phối cảnh tạo cảm giác chiều sâu thế nào |
| Độ phức tạp màu | low, medium, high | Ít hay nhiều màu/tương phản màu |
| Semantic clutter | low, medium, high | Ít hay nhiều vật thể/chi tiết có thể nhận dạng |

Các tags low/medium/high ban đầu là annotation có ví dụ neo, chưa phải kết quả từ ngưỡng tự động. Không đồng nhất gradient magnitude với độ khó đối với con người.

Input generator vẫn là ảnh raster 2D dù nó mô tả cảnh 3D. Hiện chưa cần mesh, depth map hay multi-view. Nếu sau này cần tính nhất quán qua nhiều góc nhìn, đó là một bài toán mở rộng riêng.

## 3. Danh mục background để lựa chọn

### 3.1. Sáu nền nên thu trước

| ID | Nền cụ thể nên tìm | Biểu diễn ưu tiên đợt đầu | Chi tiết mục tiêu | Câu hỏi nghiên cứu |
|---|---|---|---|---|
| B01 | Tường sơn/trát gần đồng màu, ít vết | Ảnh chụp | Low | Object có bị giống sticker khi không có nhiều texture để hòa không? |
| B02 | Cát hoặc bề mặt hạt nhỏ, ít vật thể riêng | Ảnh chụp | Medium | Texture hạt giúp hòa appearance thế nào? |
| B03 | Lá cây/cỏ dày, nhiều cạnh chồng nhau | Ảnh chụp | High | Structure có bị texture nền lấn át không? |
| B04 | Vân gỗ, đường sóng hoặc đường sọc có hướng | Ảnh chụp vân gỗ | Medium | Texture có hướng tương tác với outline object ra sao? |
| B05 | Đá/sỏi có khối, ánh sáng và bóng rõ | Ảnh render 3D có nguồn xác định | Medium–high | Có tạo halo/lệch cảm giác khối hoặc ánh sáng không? |
| B06 | Pattern hình học phẳng, lặp đều, không chữ | Pattern 2D | Medium–high | Blending có phá quy luật lặp hoặc làm mất hình object không? |

Nếu không có B05 render 3D, dùng ảnh chụp đá có bóng để thử cảm giác khối và ghi đúng `photograph`; chưa được báo đã phủ rendered_3d.

Sáu ảnh này chỉ giúp khám phá khả năng pipeline. B05 và B06 khác cả kiểu biểu diễn lẫn nội dung, nên không dùng so sánh chúng để kết luận “3D khó hơn 2D”.

### 3.2. Background mở rộng sau pilot

| Nhóm | Ví dụ | Tác dụng khi bổ sung |
|---|---|---|
| Nền phẳng cực ít chi tiết | Màu trơn, gradient nhẹ, giấy sạch | Kiểm tra giới hạn integration khi thiếu texture |
| Texture tự nhiên hạt nhỏ | Đất, sỏi mịn, tuyết hạt | Thêm biến thể của texture không đều |
| Texture tự nhiên có hướng | Vỏ cây, cỏ, gợn nước | Thử hướng texture và continuity |
| Texture hữu cơ dày | Rêu, địa y, lá khô, hoa nhỏ | Nhiều cấu trúc dễ che/lấn object |
| Vật liệu nhân tạo | Vải, thảm, bê tông, gạch | Kiểm tra chuyển từ thiên nhiên sang vật liệu khác |
| Pattern có quy luật | Caro, lưới, chấm bi, mosaic | Dễ quan sát lỗi phá nhịp/pattern |
| Pattern trừu tượng | Marble, noise nhiều tần số, đường uốn lặp | Thử cấu trúc ngoài ngữ nghĩa ảnh chụp |
| Bề mặt phản chiếu | Kim loại, kính, nước phản quang | Thử highlights và ánh sáng; thêm sau vì khó phân tích |
| Cảnh có bố cục | Góc phòng, đường phố, rừng có phối cảnh | Kiểm tra local ROI trong scene nhiều vật thể |
| Cảnh stylized | Tranh giấy cắt, flat illustration, low-poly render | Kiểm tra style khác domain pilot |

Không cần thu toàn bộ nhóm ngay. Mỗi lần mở rộng chọn một yếu tố cần kiểm tra và giữ đủ metadata để phân tích failure.

## 4. Sáu subject images nên thu trước: 4 động vật + 2 chân dung

Chia thành hai nhóm nghiên cứu: động vật và chân dung. Pilot sáu ảnh đề xuất dùng bốn ảnh động vật và hai chân dung của hai người khác nhau; đây là kiểm tra khả thi, không phải quy mô đủ để kết luận cho từng nhóm.

| ID | Object gợi ý | Yêu cầu hình ảnh | Đặc trưng kiểm tra |
|---|---|---|---|
| O01 | Mèo nhìn ngang | Tai, đầu, thân và đuôi đọc rõ | Hình hữu cơ quen thuộc, thân khối |
| O02 | Thằn lằn | Đủ thân, chân và đuôi; không quá nhỏ | Thân dài, chi tiết mảnh |
| O03 | Bướm nhìn chính diện | Hai cánh tách rõ, toàn bộ trong khung | Đối xứng và nhiều đường biên |
| O04 | Cá nhìn ngang | Thân, vây và đuôi tách rõ | Thân thuôn và các vây nhỏ |
| O05 | Chân dung chính diện | Một người, mặt rõ, nền sạch, không che mắt/mũi/miệng | Cấu trúc khuôn mặt và các chi tiết nhỏ |
| O06 | Chân dung góc 3/4 của người khác | Mặt đủ lớn, không crop mất phần quan trọng | Góc mặt, bất đối xứng phối cảnh và đường bao tóc/mặt |

Đây là danh sách thao tác để bắt đầu, không phải sáu lớp bắt buộc. Có thể thay bằng object bạn quan tâm nếu giữ đa dạng hình thái; ví dụ cá thay thằn lằn không hoàn toàn tương đương vì khác chi tiết chân/đuôi, cần ghi thay đổi.

### 4.1. Style subject nên dùng

Đợt đầu ưu tiên **một kiểu input mà pipeline hiện có xử lý ổn**, ví dụ silhouette/ảnh object tách nền có đường bao rõ. Giữ preprocessing/style nhất quán trong từng nhóm. Với chân dung, ưu tiên ảnh mặt rõ có chi tiết bên trong; silhouette chỉ có đường bao đầu không đủ kiểm tra cấu trúc mặt hoặc danh tính. Nếu dữ liệu thực tế chủ yếu là ảnh chụp object trên nền sạch, có thể dùng style đó cho cả bộ.

Không yêu cầu đổi tất cả ảnh sang silhouette nếu use case cần ảnh tự nhiên. Khi chọn style nào, báo phạm vi đó; sau audit mới thêm style còn lại như một nhánh đánh giá riêng.

**Loader hiện bỏ alpha khi convert RGB.** Vì vậy lưu original alpha/cutout trước, composite lên xám RGB(128,128,128), crop/pad giữ tỷ lệ rồi mới đưa vào generator. Alpha không tự trở thành generation mask. Nếu normalized input vẫn rò HED mask, dùng alpha support có QA; thiếu support mới dùng `ZhengPeng7/BiRefNet` offline theo PRD §6.1. Mọi methods phải chung preprocessing version.

### 4.2. Những object chưa nên đưa vào đợt đầu

- Object rất nhỏ hoặc bị crop mất bộ phận chính.
- Nhiều object dính nhau, khó biết đâu là target.
- Object đòi đọc chữ/số để nhận dạng, như logo hoặc biển báo có text.
- Object trong suốt/phản chiếu mạnh hoặc hình dạng rất mơ hồ nếu pipeline chưa xử lý được inputs dễ hơn.

Không loại vĩnh viễn các nhóm này nếu chúng thuộc mục tiêu dự án; để phase mở rộng và ghi rõ chưa được hỗ trợ/đánh giá.

### 4.3. Khi mở rộng, bao phủ riêng động vật và chân dung

Sáu ảnh đầu chỉ là smoke pilot; bốn ảnh động vật và hai chân dung chưa đại diện cho từng nhóm. Khi mở rộng, lấy nhiều ảnh/pose/nguồn khác nhau cho các nhóm: thú bốn chân, bò sát thân dài, chim, cá, côn trùng có cánh và lưỡng cư. Có thể thêm rắn hoặc động vật nhiều chân nếu phù hợp phạm vi bài. Đây là nhóm lấy mẫu thực dụng, không phải phân loại sinh học loại trừ nhau.

Theo dõi kích thước chi tiết, mức che khuất, pose, style và texture nguồn. Claim nhận dạng ở cấp nhóm hay loài phải phù hợp nhãn đánh giá. Background vẫn có thể là thiên nhiên, tường, vật liệu nhân tạo, pattern 2D và ảnh render 3D; giới hạn subject là động vật/chân dung không buộc background chỉ là môi trường tự nhiên.

Với chân dung, mở rộng theo nhiều người, góc mặt, biểu cảm, tóc, sắc độ da và điều kiện ánh sáng. Ghi metadata phục vụ độ phủ phù hợp, không suy đoán các thuộc tính nhạy cảm. Tách source/identity groups giữa dev/val/test nếu claim generalization sang người chưa thấy; không chia hai ảnh của cùng một người qua dev/test rồi gọi đó là unseen identity.

## 5. Ma trận ghép 18 dev pairs

Với 6 objects × 6 backgrounds, không cần chạy hết 36 tổ hợp. Ma trận sau cho mỗi object gặp 3 backgrounds, mỗi background gặp 3 objects.

| Object \ Background | B01 Tường | B02 Cát | B03 Lá dày | B04 Vân gỗ | B05 Đá 3D | B06 Pattern 2D |
|---|---|---|---|---|---|---|
| O01 Mèo | — | **P02 audit** | — | P03 | — | P01 |
| O02 Thằn lằn | — | P04 | P05 | — | P06 | — |
| O03 Bướm | — | — | P07 | P08 | — | P09 |
| O04 Cá | P10 | — | — | P11 | P12 | — |
| O05 Chân dung chính diện | **P13 audit** | P14 | — | — | P15 | — |
| O06 Chân dung góc 3/4 | P17 | — | **P16 audit** | — | — | P18 |

```text
P01 = O01 + B06    P02 = O01 + B02    P03 = O01 + B04
P04 = O02 + B02    P05 = O02 + B03    P06 = O02 + B05
P07 = O03 + B03    P08 = O03 + B04    P09 = O03 + B06
P10 = O04 + B01    P11 = O04 + B04    P12 = O04 + B05
P13 = O05 + B01    P14 = O05 + B02    P15 = O05 + B05
P16 = O06 + B03    P17 = O06 + B01    P18 = O06 + B06
```

Ma trận này có 12 animal pairs và 6 portrait pairs. Đây là tỷ lệ pilot, không phải trọng số chuẩn cho benchmark cuối. Việc tái sử dụng nguồn ảnh là có chủ đích để khảo sát effects trong dev. 18 pairs không phải 18 subjects hoặc 18 backgrounds độc lập. Không random split các cặp này để claim generalization sang nguồn ảnh mới.

Đợt đầu tránh backgrounds có sẵn động vật mục tiêu để recognition không bị nhầm với subject đã có trong nền. Có thể thêm trường hợp này sau như stress test, với protocol kiểm tra vị trí target.

## 6. Ba cặp chạy trước và ngân sách

### Vòng 1 — audit 21 renders

1. **P02: mèo + cát.** Quan sát giữ silhouette và hòa texture hạt.
2. **P13: chân dung chính diện + tường ít texture.** Quan sát mắt/mũi/miệng, chuyển tiếp mặt–nền và nguy cơ sticker.
3. **P16: chân dung góc 3/4 + lá dày.** Quan sát hình học mặt và việc texture nền che/méo các bộ phận mặt.

Mỗi pair chạy bảy configs ở [bảng audit](GOLDEN_SET_GUIDE.md#7-bảng-7-cấu-hình-cho-mỗi-pair-audit), cùng seed/prompt/layout/preprocessing trong pair. Tổng 21 renders, khoảng 105 giây phần render nếu mỗi lượt đúng 5 giây. Chưa tính loading, QA, metrics và annotation.

Nếu một pair không đạt với cả 7 configs, ghi `no_accept_found_in_audit`. Điều này không chứng minh cả lớp object/background không thể xử lý, và không phải lý do tự động bỏ pair.

### Vòng 2 — khảo sát default trên 15 pairs còn lại

Nếu pipeline/logging ổn, chạy default một lượt cho mỗi pair còn lại. Tổng cộng hai vòng là **36 renders**; phần render ước tính 180 giây ở mức 5 giây/lượt.

Con số 36 là 21 audit outputs + 15 default outputs, không phải chạy toàn bộ 36 tổ hợp. Ba default của audit đã được tính, không render lại nếu cùng version/seed.

### Vòng 3 — chọn chỗ cần khảo sát thêm

Dựa trên failures, chọn thêm dev pairs để mở 7-config audit hoặc manual tuning. Không cần lập tức chạy 18 × 7 = 126 renders. Với 5 giây/lượt, 126 lượt chỉ khoảng 10,5 phút phần render nhưng chi phí annotation lớn hơn; ưu tiên output mang lại thông tin mới.

Không chọn thêm chỉ từ những pair đang đẹp. Giữ coverage cho failures trên nền phẳng, nền dày và động vật có chi tiết mảnh.

## 7. Từ outputs mới tạo thành bộ ảnh mẫu

Sau audit, chọn trong dev outputs:

- 1–2 ảnh đạt rõ nếu có.
- 1 ảnh quá nổi/sticker.
- 1 ảnh subject quá chìm/mất.
- 1 ảnh lỗi cấu trúc hoặc nền bị phá.
- 1 ảnh phân vân.

Đây mới là thời điểm tạo R01–R06 theo [hướng dẫn golden set](GOLDEN_SET_GUIDE.md). Không đòi đủ loại lỗi và không ép có ảnh đạt. Nếu tất cả outputs chưa đạt, chọn best-available và ghi thiếu điều gì.

Trong cùng một pair, nếu có outputs từ quá nổi → gần đạt → quá chìm thì rất hữu ích để xây rubric. Giữ toàn bộ render logs; bộ ảnh neo là subset để giải thích, không thay thế dữ liệu đầy đủ.

## 8. Prompt và label cho động vật/chân dung

Code hiện có `prompt.subject_name` và placeholder `{animal}`. Điền nhãn subject theo metadata của từng input, kiểm tra đường format và luôn lưu prompt đã resolve. Không suy nhãn từ tên file nếu tên file không đáng tin; không cần đổi diffusion architecture.

Template hiện có “hidden”, “silhouette”, “optical illusion”; negative prompt có “cartoon”. Những từ này có thể không hợp mọi mục tiêu/style. Trước benchmark, thử một template trung tính trên dev rồi chốt dùng chung.

Ví dụ nội dung prompt đề xuất để pilot, chưa được kiểm chứng:

```text
a recognizable {animal} form seamlessly integrated into the background,
coherent shape, colors and textures consistent with the surrounding scene
```

Giữ `{animal}` để tương thích template hiện tại. Trường metadata `object_label` là tên kỹ thuật chung; giá trị trong benchmark này là nhãn động vật hoặc human portrait. Placeholder `{animal}` hiện chỉ là tên kỹ thuật; có thể điền một cụm mô tả chân dung, không cần sửa diffusion architecture.

Label gợi ý:

```text
O01 cat
O02 lizard
O03 butterfly
O04 fish
O05 human face, frontal portrait
O06 human face, three-quarter portrait
```

Tên tiếng Anh cho generation có thể cụ thể hơn nhãn chấp nhận khi human evaluation. Ví dụ cần người xem nhận ra “chim”, hay đúng một loài chim cụ thể? Cần chốt `accepted_labels` trước khi chấm, không nới nhãn sau khi thấy model sai.

Không đồng thời search prompt và ba tham số trong main experiment. Nếu đổi template trong pilot, ghi version và áp dụng cho tất cả methods; không quy cải thiện từ prompt cho agent config search.

CLIP hiện dùng câu `a photo of a {subject_name}`. Với pattern/illustration, cần kiểm tra độ phù hợp trên dev; chưa coi raw CLIP score giữa style khác nhau là thước đo recognition đã calibration.

### 8.1. Tiêu chí chân dung cần tách rõ

| Tiêu chí | Động vật | Human portrait |
|---|---|---|
| Nhận dạng cơ bản | Đúng loại/nhóm động vật đã chốt | Nhận ra khuôn mặt người ở lượt nhìn đầu |
| Cấu trúc | Thân, chi, cánh, đuôi và đặc trưng liên quan | Vị trí/tỷ lệ mắt, mũi, miệng, outline mặt; phù hợp góc nhìn |
| Integration | Appearance hòa nền, không sticker | Appearance hòa nền nhưng mặt vẫn đọc được, không seam/halo rõ |
| Bảo toàn nguồn | Không mặc định giữ texture/identity cá thể | Không mặc định giữ đúng danh tính người nguồn |

**Đã chốt: không yêu cầu giữ đúng danh tính người nguồn.** Chỉ đánh giá khuôn mặt dễ nhận ra, cấu trúc/hướng mặt hợp lý và hòa nền đẹp. Không thêm identity similarity vào reward, không reject chỉ vì output trông khác người nguồn và không claim identity-preserving generation.

VLM/CLIP với nhãn “human face” chỉ là proxy cho face recognizability, không thay thế face-structure assessment. Face detector cũng không bảo đảm mặt đẹp/hợp lý khi texture biến đổi. Audit dùng manual face-structure/integration notes; không xây face recognition hoặc identity scoring pipeline.

Báo acceptance và failure riêng cho động vật/chân dung; có thể báo trung bình hai nhóm với trọng số công bố, không chỉ một điểm gộp che khuất nhóm kém. Các tags chung giữ nguyên, bổ sung `face_structure_error`; `identity_drift` không là failure trong phạm vi hiện tại.

## 9. Layout và local background quan trọng hơn tên loại nền

Một ảnh toàn cảnh có nhiều lá nhưng vùng đặt object là khoảng trời trống sẽ không phải case “nền lá dày” ở nơi cần blend. Gắn tags cho cả ảnh tổng thể và vùng đặt subject.

Đợt đầu:

- Đặt object trong ảnh, không cắt bộ phận quan trọng.
- Đảm bảo các chi tiết cần nhận dạng còn nhìn được ở resolution cuối.
- Chọn quy tắc kích thước nhất quán, ví dụ khoảng bounding-box hợp lý được QA, thay vì mỗi output đẹp thì tự thay scale.
- Freeze layout theo pair trước search; không dùng generation mask thay cho evaluation region đã review.
- Lưu edge maps trước khi quy lỗi output cho optimizer.

Với nền phẳng, natural integration có thể dựa vào màu, shading và biên chuyển tiếp; không đòi appearance giống lá/đá khi background không có texture đó. Với nền pattern đều, xem continuity của pattern trong và quanh object, nhưng vẫn giữ tiêu chí nhận dạng.

## 10. Nếu muốn nghiên cứu 2D so với 3D một cách có cơ sở

Sáu backgrounds đề xuất không đủ để tách nguyên nhân style. Cần extension có nhiều ví dụ trong từng nhóm và các nội dung tương ứng:

| Nội dung giữ gần tương ứng | Bản 2D | Bản mô tả khối/3D |
|---|---|---|
| Gạch/tiles | Pattern phẳng | Render tiles có phối cảnh/bóng |
| Đá | Illustration đá phẳng | Render đá có khối |
| Lá | Pattern lá flat | Cảnh lá có ánh sáng/occlusion |

Thử cùng object/layout/search policy và so theo nhóm. Nên có nhiều sources mỗi nhóm, không kết luận từ một cặp. Dù được match, ánh sáng/texture vẫn có thể khác; báo các yếu tố chưa kiểm soát được.

Đây là ablation về domain/style sau pilot, chưa cần để bắt đầu xây optimizer.

## 11. Metadata tối thiểu cho bộ input

```yaml
subject:
  id: O05
  object_label: human face, frontal portrait
  category: human_portrait
  representation: photograph
  shape_tags: [face, eyes, nose, mouth]
  identity_group: person_001
  source_group: subject_source_05
  path: REQUIRED
  source_and_license: REQUIRED

background:
  id: B01
  representation: photograph
  material_or_content: wall
  texture_density: low
  repetition: none
  depth_cues: flat
  source_group: background_source_01
  path: REQUIRED
  source_and_license: REQUIRED

pair:
  id: P13
  split: dev
  local_texture_density: low
  layout_version: REQUIRED
  prompt_version: REQUIRED
  evaluation_assets_version: REQUIRED
  status: not_rendered
```

Đây là mẫu metadata, chưa có files tương ứng. Nếu không biết nguồn hoặc style thì ghi unknown; không đoán đó là ảnh render 3D chỉ vì trông có chiều sâu.

## 12. Thu ảnh từ đâu và cần lưu gì?

Bạn có thể dùng ảnh tự chụp/tự vẽ, assets hiện có được phép sử dụng, hoặc bộ ảnh công khai có điều kiện sử dụng phù hợp. Chưa cần chọn dataset lớn hay tải hàng nghìn ảnh.

Với backgrounds, ảnh tự chụp tường, cát, lá và vân gỗ là cách kiểm soát source tốt. Pattern hình học có thể tự thiết kế và lưu thông số tạo. Với rendered_3d, dùng asset có provenance rõ; nếu chưa có thì đánh dấu nhóm này đang thiếu.

Lưu bản gốc, đường dẫn nguồn, author/license nếu có, và mọi bước crop/resize. Không dùng biến thể crop của cùng một ảnh làm hai nguồn độc lập. Hướng dẫn này chưa tìm, tải hoặc xác minh license của ảnh cụ thể nào.

## 13. Split và bước tiếp theo

18 pairs ở mục 5 đều tái dùng 6 subjects và 6 backgrounds, nên toàn bộ thuộc dev. Không split ma trận này thành dev/val/test.

Sau pilot, mở dev lên 120 pairs (60 animal/60 portrait), thu 30 validation pairs (15/15) và 100 locked test pairs (50/50) từ source groups tách biệt. Đây là planning target thống nhất, chưa phải dataset đã có. Coverage quotas chi tiết theo representation×local texture ở PRD §15; không có đủ nguồn thì báo pilot, không âm thầm thay benchmark.

Nếu chỉ có nguồn lặp, có thể kiểm tra unseen combinations, nhưng phải gọi đúng setting. Muốn claim unseen object categories còn cần giữ lại cả category, không chỉ file ảnh mới cùng category.

### Checklist việc làm ngay

- [ ] Thu O01–O06: bốn động vật + hai chân dung khác người; QA style/preprocessing theo nhóm.
- [ ] Thu B01–B06, đánh dấu rendered_3d nếu chưa có nguồn xác thực.
- [ ] Tạo inventory và source groups; chưa cần gán đẹp/xấu cho input.
- [ ] QA P02, P13, P16: label/prompt, alpha composite, crop/pad giữ tỷ lệ, edges, mask và layout.
- [ ] Nếu M1 rò: alpha support hoặc BiRefNet offline; review tóc/tai/chi mảnh, freeze preprocessing version và rerun affected comparisons.
- [ ] Render 7 configs/pair, lưu đầy đủ 21 outputs.
- [ ] Chấm và chọn ảnh neo từ outputs mới, giữ uncertainty.
- [ ] Chạy default 15 pairs còn lại nếu QA ổn.
- [ ] Phân tích failure rồi mới mở rộng dữ liệu hoặc thêm agent.

Tài liệu tiếp theo: [Golden set và annotation](GOLDEN_SET_GUIDE.md), [PRD và kiến trúc](PRD_CAMOUBLEND.md).
