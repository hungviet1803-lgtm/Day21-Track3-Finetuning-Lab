# Reflection — Lab 21

**Nguyễn Việt Hùng — 2A202602972**

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Cổng hồi quy đánh trượt một model đạt 0.970 trên tác vụ. Tôi nghĩ phần khó là làm
fine-tune thắng prompt tối ưu; hoá ra phần đó dễ (+0.205), còn cái giá nằm ở chỗ tôi không
nhìn: năng lực trả lời câu hỏi phổ thông giảm từ 0.791 xuống 0.478. Ngạc nhiên thứ hai là
`wrong_lr`: chỉ đổi learning rate từ 1e-4 xuống 1e-5 mà target từ 0.970 về 0.000 — nút
"nhàm chán" nhất lại là nút quyết định nhất, còn rank (tăng 17 lần, 16 → 283) gần như không
đổi gì.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở train, mà ở vận hành Colab. Tôi chạy nhầm với `EVAL_LIMIT=8` vì không để ý giá
trị mặc định của ô, phải chạy lại NB2 + NB5. Khi tải kết quả về lần đầu, gợi ý tự động của
Gemini trong Colab sửa lệnh `zip` thành chỉ nén `results/`, nên file thiếu toàn bộ adapter
mà tôi không biết cho tới khi mở ra. Tôi cũng mở nhầm giữa tab notebook gốc và tab bản sao —
hai tab chạy trên hai máy ảo khác nhau. Tôi dự đoán phần khó là cấu hình LoRA; thực tế phần
tốn thời gian là bảo đảm mình đang chạy đúng thứ, trên đúng dữ liệu, ở đúng chỗ.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi tin train loss thấp nghĩa là model tốt, và fine-tune một tác vụ hẹp thì không ảnh hưởng
gì tới phần còn lại của model. Cả hai đều sai trong số đo của tôi: `attn_only` có loss thấp
nhất nhưng không tốt hơn `correct`, và train 100% trên một dạng dữ liệu làm model mất khoảng
40% năng lực phổ thông chỉ sau 30 step. LoRA không tự bảo vệ model khỏi quên nếu dữ liệu chỉ
có một loại.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code để đọc repo, chạy NB1 và bộ test trên máy (không GPU), hướng dẫn từng
bước trên Colab, kiểm chéo số liệu trong `results/` và viết bản nháp report. Chỗ nó sai hoặc
làm tôi mất thời gian: ban đầu nó tự sửa khá nhiều code pipeline (thêm script chọn prompt
(b), thí nghiệm replay, đổi `max_length`) khi tôi chưa yêu cầu, và khi Colab in chữ đỏ tôi
tưởng là lỗi nên đã bắt hoàn tác toàn bộ — thật ra đó chỉ là output của `git diff`. Nó cũng
định tải model 9 GB về máy tôi dù máy không chạy nổi, tôi phải dừng lại. Gemini trong Colab
thì sửa lệnh nén file làm mất adapter. Bài học: AI tăng tốc rất nhiều, nhưng tôi vẫn phải
đọc từng lệnh trước khi chạy và tự quyết định cái gì được thay đổi.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Dựng tập eval và đo baseline trước khi động vào training: một tập target chấm khách quan,
một tập năng lực chung mà khách hàng không được phép mất, và prompt tốt nhất tôi viết được —
rồi đóng băng cả ba. Cụ thể, tôi sẽ hỏi khách hàng model còn phải làm những việc gì khác
ngoài tác vụ mới, vì đó chính là thứ lab này cho thấy dễ mất nhất. Chỉ khi prompt tốt nhất
không đủ, tôi mới fine-tune — và trộn sẵn 1–5% dữ liệu phổ thông ngay từ lượt đầu.
