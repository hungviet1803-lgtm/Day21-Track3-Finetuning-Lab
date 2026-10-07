# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Việt Hùng  **MSSV**: 2A202602972  **Ngày**: 2026-10-08
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (Colab Free, 14,6 GB khả dụng) — NB1 + test chạy thêm trên CPU cá nhân

> Mọi con số dưới đây lấy từ `results/` (đã kiểm chéo). Lượt chạy: `colab/Lab21_RUN_ALL.ipynb`
> trên commit `d27c1c0`, code của lab không sửa, `EPOCHS=2`, eval đầy đủ (50 target + 15 regression).

---

## Tóm tắt một dòng

Fine-tune **thắng** prompt tối ưu trên tác vụ (target **0.970** vs **0.765**, +0.205) nhưng
**làm hỏng năng lực phổ thông** (regression **0.478** vs **0.791**, −0.313). Cổng hồi quy:
**FAILED**. Không nên deploy bản này như một model dùng chung.

---

## 0. Lựa chọn thí nghiệm và lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định tier T4) | Lớn nhất vừa Colab Free T4 với LoRA 16-bit (đo được peak 8,78 GB / 14,6 GB). Không mặc định QLoRA vì nhà cung cấp khuyến nghị không lượng tử hoá dòng model này — 4-bit được *đo* ở run `qlora`. Máy cá nhân chỉ có MX550 2 GB nên không train tại chỗ. |
| Dataset | 250 ticket CSKH tiếng Việt → JSON 4 trường (corpus mặc định) | Mọi nhóm điểm có thang khách quan (so nhãn, parse JSON, keyword) — không cần LLM judge, không có "điểm cho không". Chạy corpus mặc định để có mốc so được với số đo đã công bố của lab. |
| Prompt (b) | `OPTIMIZED_PROMPT` gốc, không sửa (SHA `719e74d3b6232053`) | Giữ nguyên để phép so sánh kiểm chứng được. Giới hạn của lựa chọn này: xem §3. |
| `MASK_MODE` | `assistant-only` | Corpus không có trace suy luận nên `masked-think`/`response-only` cho mask y hệt. |

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket → JSON triage; eval: 50 target + 15 regression |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (giá trị tier) — p95 đo được là **98**, p99 100, max **101**, gợi ý **256** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / **30** (= ⌈225 / 16⌉ × 2; batch hiệu dụng 1 × 16) |
| Độ chính xác | fp16 (T4 là Turing, không có bf16) |

**Vì sao giữ 1024 dù p95 gợi ý 256.** Mẫu dài nhất là 101 token, nên ở cả 1024 lẫn 256
**không mẫu nào bị cắt** — hai giá trị cho cùng một tập huấn luyện. Với
`per_device_batch=1` không có padding giữa các chuỗi, nên `max_length` chỉ là trần cắt,
không làm tăng tính toán hay VRAM. Đặt 256 đúng tinh thần "đo rồi mới đặt" hơn nhưng
không đổi kết quả; tôi giữ cấu hình gốc để không thêm biến vào phép so sánh.

**Template có giữ khối `<think>` không?** **Có** — `results/template_check.json`:
`"verdict": "reasoning preserved — safe to train on traces"`, nội dung
`buoc 1: kiem tra. buoc 2: tra loi.` còn nguyên sau `apply_chat_template`.
Corpus không có trace, nên mỗi mẫu train mang một khối `<think>\n\n</think>` rỗng. Lúc eval,
`generate_batch(enable_thinking=False)` gửi đúng tiền tố đó (tôi kiểm ở mức token: prompt
eval là tiền tố chính xác của chuỗi train), nên model lúc eval thấy cùng ngữ cảnh như lúc
train. `valid_trace_rate = 0.0` ở NB5 là đương nhiên, không phải dấu hiệu sụp suy luận:
không có trace nào để giữ.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (giải mã ngược các vị trí `labels != -100`, `results/mask_proof.json`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị che: `<|im_start|>system\nPhân loại ticket sau.<|im_end|>`, toàn bộ ticket, và
`<|im_start|>assistant\n<think>\n\n`. Đối chứng `MASK_MODE=everything`: 94/94 token (100%)
bị tính loss — model sẽ học viết lại câu hỏi. `<|im_end|>` nằm trong loss là đúng: đó là
tín hiệu dừng.

**Vì sao không dùng `assistant_only_loss` của TRL** (`scripts/check_mask_agreement.py`):
template của Qwen3.5 không có marker `{% generation %}`, nên mask phía tokenizer của TRL
**rỗng — 0/31 token**, chỉ kèm một cảnh báo. Mask của lab cho 11/31 token, đúng phần trả
lời. Tin cờ thư viện thì run sẽ train trên không token nào mà không báo lỗi. NB3 vì thế nạp
dữ liệu đã token hoá sẵn bằng chính mask đã chứng minh ở trên.

---

## 3. Ba baseline (NB2) và bản fine-tune

*(results/baselines_frozen.json, results/verdict.json — n = 50 target, 15 regression)*

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3353.9 |
| (b) base + optimized prompt | **0.765** | 0.791 | 1.000 | **1059.8** |
| (c) LoRA fine-tune (`correct`) | **0.970** | **0.478** | 1.000 | 1413.8 |

**(b) có thật sự mạnh hơn (a) không?** Có, rất rõ: 0.000 → 0.765 target, 0.000 → 1.000
format, và **nhanh hơn 3,2 lần** (3354 → 1060 ms). Prompt ngây thơ khiến model trả lời bằng
văn xuôi tới trần 160 token, nên vừa sai định dạng vừa chậm; prompt (b) buộc model in đúng
một object JSON rồi dừng. Regression của (a) và (b) bằng nhau (0.791) vì nhóm regression
không dùng system prompt — đó là cùng một base model.

**Có sửa `OPTIMIZED_PROMPT` không?** Không (`verify`: *baseline (b) prompt unmodified*).

**Giới hạn của mốc (b), nói thẳng.** Trên tập train, mỗi cụm từ ánh xạ tới **đúng một
nhãn** (22 cụm intent, 11 urgency, 10 sentiment), và vài quy ước ngược nghĩa thông thường
("hoàn lại" → `doi_tra`, "quá hạn rồi" → `cao`, "mình vẫn tin tưởng shop" → `tich_cuc`).
Prompt (b) chỉ có 1 ví dụ nên không thể biết các quy ước này. Trong một lượt thử nghiệm
riêng (code đã hoàn tác, số **không** nằm trong `results/`), tôi chấm thử trên tập val
25 mẫu: prompt gốc 0.65, một bản 8 ví dụ few-shot 0.74. Tức là (b) gốc chưa phải mốc mạnh
nhất có thể. Khoảng cách +0.205 đủ lớn để tôi tin fine-tune vẫn thắng một prompt tốt hơn
về target, nhưng biên thật sẽ nhỏ hơn con số trong bảng.

**Thứ tự đo — khai báo.** Lượt đầu tôi chạy nhầm với `EVAL_LIMIT=8` (giá trị mặc định của ô
Colab), nên NB2 chỉ chấm 8 mẫu và `verify` báo *full eval set used: FAIL*. Tôi chạy lại
`nb2 nb5 nb6` với eval đầy đủ, dùng lại adapter đã train. Như vậy số (a)/(b) trên 50 mẫu
được đo **sau** khi train. Điều này không mở đường cho thiên lệch: prompt (b) và tập eval
không đổi (SHA và checksum đều được `verify` kiểm), và tôi không chỉnh gì giữa hai lượt.
Lượt 8 mẫu cũng cho một bài học riêng: nó báo regression Δ −0.125, còn trên đủ 15 câu là
**−0.313** — lượt rút gọn đánh giá thấp mức quên tới 2,5 lần.

---

## 4. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)

*(results/runs.csv, results/autopsy.json — cả bốn run 30 step, cùng seed, cùng mask)*

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | latency ms | s train | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 loại module) | 16 | 32,464,896 | 1e-4 | 0.6257 | **0.970** | 1.000 | 1413.8 | 412.9 | 8.78 |
| `attn_only` | q,v | **283** (matched) | 32,456,704 | 1e-4 | **0.5383** | 0.965 | 1.000 | **921.6** | 272.1 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | **1e-5** | 1.5702 | **0.000** | 0.000 | 5366.7 | 402.2 | 8.78 |
| `qlora` | text-linear, **4-bit** | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1.000 | 1839.3 | 468.8 | **3.86** |

Biến duy nhất mỗi run đổi so với `correct`: `attn_only` — **vị trí** (rank nâng lên để giữ
ngân sách, lệch 0,025%); `wrong_lr` — **learning rate** (÷10); `qlora` — **độ chính xác
của base** (4-bit NF4).

Xếp hạng theo **target**: `correct` 0.970 ≈ `attn_only` 0.965 > `qlora` 0.940 ≫ `wrong_lr`
0.000. Xếp hạng theo **train loss**: `attn_only` < `correct` < `qlora` < `wrong_lr`. Hai
thứ tự **khác nhau ở vị trí đầu**.

**4.1 — `attn_only` vs `correct`.** Cùng ngân sách 32,46 M tham số, `attn_only` đạt 0.965
so với 0.970 — chênh 0.005, tức **một trường sai trên 200** trường được chấm. Đó là hoà,
không phải thắng thua. Nhưng train loss của `attn_only` *thấp hơn* rõ (0.538 vs 0.626):
nếu xếp hạng bằng loss, tôi sẽ kết luận "attention-only với rank cao là tốt hơn" và chọn
nhầm Lỗi #1. Loss thấp hơn ở đây không mua được năng lực tác vụ nào; với 225 mẫu ngắn và
rank 283 dồn vào ít lớp, nó trông giống khớp dữ liệu train hơn là học tốt hơn. Về *vị trí
vs rank*: rank tăng 17,7 lần (16 → 283) **không** đem lại gì trên target, nên rank không
phải đòn bẩy ở bài này. Vị trí cũng không tạo khác biệt về chất lượng — vì tác vụ quá hẹp
(một bảng tra ~43 cụm từ), cả hai cách đặt adapter đều đủ dung lượng. Khác biệt thật nằm ở
**chi phí suy luận**: `attn_only` nhanh hơn 35% (922 vs 1414 ms) và train nhanh hơn 34%
(272 vs 413 s), vì adapter chưa merge thêm phép nhân ở mỗi module được gắn, và q,v chỉ có
ở các lớp full-attention của kiến trúc lai Qwen3.5 (2 loại module thay vì 12). Kết luận
đúng từ số đo của tôi là "trên tác vụ này, vị trí không đổi chất lượng nhưng đổi chi phí",
**không** phải "text-linear thắng" như deck dự đoán.

**4.2 — `wrong_lr`.** Chỉ đổi LR 1e-4 → 1e-5, train loss cuối là **1.570** so với 0.626
(gấp 2,5 lần) — model gần như chưa rời khỏi base sau 30 step. Hậu quả trên tác vụ là tuyệt
đối: **target 0.000, format 0.000**, và latency 5367 ms — model trả lời bằng văn xuôi tới
trần token như base với prompt ngây thơ (a). Nếu chỉ nhìn loss mà không biết LR, tôi sẽ
kết luận "LoRA học kém / dữ liệu không đủ / cần thêm rank hay thêm epoch" — tức đổ lỗi cho
phương pháp, trong khi lỗi nằm ở một con số. Đây là biến có ảnh hưởng lớn nhất trong cả
bốn run: 0.970 → 0.000, so với 0.005 khi đổi vị trí và 0.030 khi đổi sang 4-bit.

**4.3 — `qlora`.** Peak VRAM **3.86 GB vs 8.78 GB — giảm 56%**. Cái giá: target 0.940 vs
0.970 (−0.030, khoảng 6 trường sai thêm trên 200), train chậm hơn 13,5% (469 vs 413 s) và
suy luận chậm hơn 30% (1839 vs 1414 ms) vì phải giải lượng tử mỗi lần nhân. Số đo của tôi
**ủng hộ một phần** khuyến nghị "không dùng QLoRA cho dòng model này": có mất chất lượng
đo được, và không được gì về tốc độ — nhưng mức mất nhỏ. Trên T4, LoRA 16-bit đã vừa (8,78
/ 14,6 GB) nên không có lý do dùng 4-bit; QLoRA chỉ đáng khi VRAM thật sự không đủ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.313` (ngưỡng −0.020) · `valid_trace_rate = 0.0`

*(results/verdict.json: "general capability regressed by 0.313 (tolerance 0.020)")*

Bản fine-tune làm đúng điều nó được train để làm: target 0.970 so với 0.765 của prompt tốt
nhất, format vẫn tuyệt đối, và prompt lúc dùng chỉ còn một câu `"Phân loại ticket sau."`.
Nhưng trên 15 câu hỏi phổ thông, keyword recall rơi từ 0.791 xuống 0.478 — mất khoảng 40%
năng lực vốn có, gấp 15 lần ngưỡng cho phép. Nguyên nhân nằm ở dữ liệu, không ở cấu hình
LoRA: 100% trong 225 mẫu train là cùng một dạng "ticket → JSON", không có một mẫu nào dạy
model rằng còn có loại câu hỏi khác. Sau 30 step, model học được quy tắc "mọi đầu vào đều
cần ra JSON triage" mạnh hơn mức nó giữ được hành vi trả lời bình thường. Câu trả lời thực
tế của model cho 15 câu này không được lab lưu lại, nên tôi không khẳng định được nó trả
lời bằng JSON hay sai kiến thức; lần chạy mô phỏng trên model 0.8B của lab
(`SIMULATION-FINDINGS.md`) ghi nhận dạng thứ nhất. Phán quyết FAILED vì thế là đúng: một
model chỉ làm được triage sẽ phá vỡ mọi chức năng khác nếu dùng làm model chung. Đây cũng
**không** phải lập luận để nới ngưỡng: biện pháp đúng là trộn 1–5% dữ liệu phổ thông vào
tập train (deck §6.3) rồi đo lại, hoặc chỉ dùng adapter cho đúng luồng triage và để base
model trả lời mọi thứ khác (NB6 cho thấy hoán đổi adapter trên cùng base là khả thi).

---

## 6. Định tính — có cả ca THUA

*(results/qualitative.json)* Trong 50 ticket, fine-tune đúng hoàn toàn **44**, và **6** ca
còn lại đều 0.75 (sai đúng một trường). Lab không lưu dự đoán từng mẫu của (b) — NB2 chỉ
ghi điểm tổng — nên cột (b) không thể điền theo từng ticket; tôi không suy đoán nó.

| # | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune | Nhận xét |
|---|---|---|---|---|
| 1 (i=2) | "…đèn bàn LED… **Hoàn tiền. Quá hạn rồi.** Cảm ơn shop nhiều." | hoan_tien · **cao** · tich_cuc | hoan_tien · cao · tich_cuc (1.00) | ✅ thắng một quy ước khó: "quá hạn rồi" = `cao`, và sentiment theo câu cảm ơn dù đang đòi tiền |
| 2 (i=0) | "…chuột không dây… **Cho tôi trả lại. Gấp.** Shop hỗ trợ tốt." | doi_tra · cao · tich_cuc | doi_tra · cao · tich_cuc (1.00) | ✅ "trả lại" (hàng) → `doi_tra`, không nhầm sang `hoan_tien` |
| 3 (i=3) | "…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều." | hoan_tien · **thap** · tich_cuc | hoan_tien · **trung_binh** · … (0.75) | ❌ **FT thua**: urgency sai |
| 4 (i=5) | "…nồi chiên không dầu… Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi." | san_pham_loi · **thap** · trung_tinh | san_pham_loi · **trung_binh** · … (0.75) | ❌ **FT thua**: urgency sai |
| 5 (i=39) | "…nồi chiên không dầu… Hoàn tiền. **Khi nào tiện.** Quá tệ." | hoan_tien · **thap** · tieu_cuc | hoan_tien · **trung_binh** · … (0.75) | ❌ **FT thua**: urgency sai |
| 6 (i=46) | "…đèn bàn LED… Sai màu. **Khi nào tiện.** Shop hỗ trợ tốt." | san_pham_loi · **thap** · tich_cuc | san_pham_loi · **trung_binh** · … (0.75) | ❌ **FT thua**: urgency sai |

**Mẫu chung ở các ca thua: có, và nó rất sắc.** Cả 6 lỗi đều là cùng một lỗi —
ticket chứa **"Khi nào tiện"** bị gán urgency `trung_binh` thay vì `thap`. Fine-tune sai
**6/6** ticket có cụm này trong eval (i = 3, 5, 12, 39, 41, 46), trong khi đúng **12/12**
ticket mang hai cụm `thap` còn lại ("Không vội" 7/7, "Hỏi cho biết thôi" 5/5). Đây không
phải thiếu dữ liệu: trong tập train, "khi nào tiện" xuất hiện **30 lần**, lần nào cũng là
`thap` — là cụm `thap` *phổ biến nhất*. Tôi chưa kiểm chứng được nguyên nhân (cần GPU để
so log-prob). Giả thuyết hợp lý nhất: "khi nào" là từ để hỏi thời gian, dễ gắn với "cần
xử lý" theo hiểu biết sẵn có của base, và cùng mở đầu một cụm intent khác ("khi nào có
tiền về"); 30 step ở LR 1e-4 chưa đủ để ghi đè cái prior đó. Điều đáng ghi nhớ hơn giả
thuyết là: toàn bộ 3% điểm còn thiếu của một model "0.970" là **một lỗi hệ thống duy nhất,
lặp lại 100%** — điểm tổng không cho thấy điều đó, chỉ đọc từng ca mới thấy. Một hệ thống
triage thật sẽ đẩy mọi ticket "khi nào tiện" lên hàng ưu tiên trung bình.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** deploy bản fine-tune này như một model dùng chung. Nó thắng
prompt tối ưu trên tác vụ một cách thật (0.970 vs 0.765, format 1.000 ở cả hai), nhưng cái
giá là mất khoảng 40% năng lực trả lời câu hỏi phổ thông, và cổng hồi quy bắt được điều đó
đúng như thiết kế. Nếu chỉ báo cáo điểm target — hoặc tệ hơn, chỉ báo cáo train loss — đây
sẽ trông như một chiến thắng rõ ràng. Có hai cách dùng được kết quả này: (1) giữ adapter
riêng và chỉ bật nó cho luồng triage, để base model xử lý mọi thứ khác — NB6 xác nhận merge
không làm tụt điểm (0.970 → 0.970) và hoán đổi adapter trên cùng một base hoạt động; hoặc
(2) train lại với 1–5% dữ liệu phổ thông trộn vào rồi đo lại cả bốn nhóm. Trước khi dùng
thật, còn phải sửa lỗi hệ thống "khi nào tiện".

**Đòn bẩy thật sự** trong lab này, theo thứ tự đo được: (1) **learning rate** — sai thang
một bậc là từ 0.970 về 0.000; (2) **thành phần dữ liệu** — chính nó gây ra FAILED, không
phải cấu hình LoRA nào; (3) **mask và căn chỉnh prompt** — không phải đòn bẩy *trong* bốn
run vì cả bốn dùng chung mask đúng, nhưng nếu tin cờ `assistant_only_loss` thì cả lab sẽ
train trên 0 token; (4) **độ chính xác 4-bit** — −0.030, đổi lấy 56% VRAM; (5) **vị trí /
rank** — −0.005, tức không đáng kể về chất lượng trên tác vụ hẹp này, chỉ khác về tốc độ.
Thứ tự đó ngược với độ nổi tiếng của các nút: rank là nút được bàn nhiều nhất, và là nút
ít quan trọng nhất ở đây.

**Ba điều tôi học được:**
1. **"Chạy xong không lỗi" khác với "chạy đúng".** Lượt đầu của tôi chạy hết NB1 → NB5 trong
   39 phút, mọi ô đều có dấu ✓ xanh, và ra luôn một phán quyết. Chỉ khi đọc kỹ output của
   `verify` tôi mới thấy dòng *full eval set used: FAIL* — ô Colab để sẵn `EVAL_LIMIT=8` và
   tôi không đổi. Điều làm tôi giật mình nhất là con số sai không trông sai: lượt 8 mẫu báo
   mức quên −0.125, nghe hợp lý, trong khi số thật là −0.313. Nếu không có cổng kiểm tra, tôi
   đã nộp một kết luận dựa trên 8 ticket. Từ giờ tôi kiểm tra `n` của mọi con số trước khi
   đọc giá trị của nó.
2. **Không xếp hạng model bằng train loss.** Trước lab tôi mặc định loss thấp hơn là model
   tốt hơn. Ở đây `attn_only` có loss thấp nhất (0.538) nhưng chỉ hoà `correct` trên tác vụ;
   còn `wrong_lr` có loss 1.570 — tôi tưởng nghĩa là "học chậm hơn", hoá ra là model không
   in nổi một object JSON nào (target 0.000). Loss chỉ có nghĩa khi đặt cạnh một chỉ số tác
   vụ đo trên dữ liệu model chưa thấy.
3. **Điểm tổng che lỗi hệ thống.** Con số 0.970 khiến tôi định ghi "gần như hoàn hảo" rồi
   sang phần khác. Đọc từng ca sai, tôi mới thấy cả 6 lỗi là cùng một cụm "Khi nào tiện",
   sai 6/6 lần, dù cụm đó có 30 lần trong dữ liệu train. Trong triage thật, đó là cả một
   nhóm khách hàng bị xếp sai mức ưu tiên — một lỗi mà không chỉ số tổng hợp nào báo ra.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) trộn ~12 câu hỏi phổ thông (≈5%), câu trả lời do
chính base model sinh, rồi chạy lại cùng 30 step để xem regression có về ngưỡng không và
target mất bao nhiêu; (2) lưu dự đoán từng mẫu của (b) và của fine-tune trên tập regression
để điền nốt cột (b) ở §6 và xem fine-tune trả lời câu hỏi phổ thông bằng gì; (3) làm (b)
mạnh hơn bằng hướng dẫn gán nhãn rút từ tập train, chọn trên tập val, để biết biên thắng
thật của fine-tune.

---

## Phụ lục — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap** — `results/merge_check.json`: trước merge 0.970, sau
  merge 0.970, Δ 0.000 (ngưỡng 0.01, n = 50). Hot-swap: NB6 nạp `correct`, `attn_only`,
  `qlora` lên cùng một base và chạy từng adapter (log trên Colab).
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
