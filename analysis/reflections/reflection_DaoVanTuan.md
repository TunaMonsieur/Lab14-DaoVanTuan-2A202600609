# Báo cáo Cá nhân (Individual Reflection) — Đào Văn Tuấn

> Lab Day 14 — AI Evaluation Factory (Expert Level)
> MSSV: 2A202600609

## 1. Đóng góp Kỹ thuật (Engineering Contribution)

Các module phức tạp tôi xây dựng/hoàn thiện trong hệ thống đánh giá:

- **Knowledge Base + Retriever ([engine/knowledge_base.py](../../engine/knowledge_base.py)):** xây corpus 5 tài liệu → 25 passages có ID ổn định, cài đặt retriever **TF-IDF + cosine similarity thuần Python** (tự tính IDF làm mượt, vector hóa, cosine) để đo Hit Rate/MRR thật.
- **Multi-Judge Consensus Engine ([engine/llm_judge.py](../../engine/llm_judge.py)):** gọi **2 judge độc lập song song** (OpenRouter `openai/gpt-oss-120b:free` + Ollama `kamekichi128/qwen3-4b-instruct-2507:latest`), tính agreement, xử lý xung đột tự động (lệch >1 điểm → lấy điểm thận trọng `min`), cache trên đĩa và **fallback** khi cloud hết quota.
- **Async Runner ([engine/runner.py](../../engine/runner.py)):** chạy song song theo batch bằng `asyncio.gather`, tích hợp Retrieval + Multi-Judge + đo latency/token/cost cho từng case.
- **Orchestrator + Release Gate ([main.py](../../main.py)):** chạy Regression V1 vs V2 và logic auto Release/Block theo 4 ngưỡng (chất lượng/retrieval/cost/safety).

**Kết quả chạy thật:** 58 cases × 2 phiên bản, Hit Rate 92.3%, Agreement 94.8%, Release Gate = APPROVE (V2 +0.31 điểm so với V1).

## 2. Chiều sâu Kỹ thuật (Technical Depth)

**MRR (Mean Reciprocal Rank):** trung bình của `1/vị_trí` của tài liệu đúng đầu tiên trong danh sách trả về. Nếu đáp án ở vị trí 1 → 1.0; vị trí 2 → 0.5; không có → 0. Khác Hit Rate (chỉ nhị phân có/không trong top-k), MRR **thưởng cho việc xếp đáp án lên cao**. Hệ thống đạt MRR 0.904 nghĩa là tài liệu đúng gần như luôn ở vị trí số 1 — retrieval rất tốt.

**Cohen's Kappa:** đo độ đồng thuận giữa 2 judge **sau khi loại trừ đồng thuận do may rủi**: `κ = (Po − Pe)/(1 − Pe)`, với Po = tỉ lệ đồng ý thực tế, Pe = tỉ lệ đồng ý kỳ vọng ngẫu nhiên. Kappa quan trọng hơn agreement thô vì 2 judge có thể "tình cờ" cùng cho điểm cao. Hệ thống: V1 κ=0.322 (yếu) → V2 κ=0.501 (trung bình) — chứng tỏ V2 cho câu trả lời rõ ràng hơn nên 2 judge dễ thống nhất hơn.

**Position Bias:** xu hướng LLM-judge thiên vị câu trả lời theo **vị trí** (A trước/B sau) thay vì theo chất lượng. Tôi cài `check_position_bias()` chấm theo 2 thứ tự đảo nhau; nếu kết luận đổi theo vị trí → judge có thiên vị, cần trung bình hóa kết quả 2 chiều.

**Trade-off Chi phí ↔ Chất lượng:** judge mạnh (cloud) cho điểm chuẩn hơn nhưng tốn quota/tiền; judge local (qwen 4B) miễn phí nhưng nhiễu hơn. Kết hợp 2 judge + cache + fallback giúp **giảm chi phí mà vẫn giữ độ tin cậy** (đo bằng Kappa). Cache khử trùng lặp (rerun = 0 đồng), và đề xuất giảm 30% chi phí: chỉ gọi judge cloud cho các case mà judge local "không chắc" (điểm gần ngưỡng), còn lại tin judge local.

## 3. Giải quyết vấn đề (Problem Solving)

Trong quá trình xây dựng, tôi gặp và xử lý nhiều vấn đề thực tế:

1. **Quota free-tier cực chặt:** Gemini 2.x báo `limit:0`, NVIDIA DeepSeek 429 liên tục, OpenRouter `gpt-oss-120b` đòi credit. → Giải pháp: trừu tượng hóa **client thống nhất** qua OpenAI SDK, dễ đổi provider; dùng bản `:free` của OpenRouter; thêm **fallback** sang model local để pipeline luôn chạy xong.
2. **Pipeline treo do nghẽn Ollama:** ban đầu dùng 2 model local cùng lúc gây tranh tài nguyên + backoff 30s vô ích. → Giải pháp: chuyển 1 judge lên cloud, **fail-fast** (bỏ sleep ở lần thử cuối, giảm retries cho cloud judge), giảm concurrency.
3. **Lỗi encoding tiếng Việt trên Windows (cp1252):** → thêm `setup_utf8()` reconfigure stdout/stderr.
4. **Model local trả JSON sai định dạng:** → viết `extract_json()` bóc JSON cân bằng ngoặc + fallback điểm trung lập khi parse lỗi.
5. **SDG bị cắt giữa chừng do hết quota ngày:** → làm SDG **resumable** (chỉ sinh passage còn thiếu, đổi model bù quota) để đạt đủ 58 cases.

**Bài học:** Giá trị lớn nhất của một Evaluation Factory là **định vị được lỗi nằm ở tầng nào** chứ không chỉ đưa ra một điểm số. Nhờ tách riêng Retrieval (Hit Rate 92%, MRR 0.90) khỏi Generation, nhóm xác định được phần lớn lỗi đến từ **Prompting/Model chứ không phải Retrieval** — điều mà nếu chỉ chấm câu trả lời sẽ không bao giờ thấy. Đồng thời, dùng **2 judge + Cohen's Kappa** cho thấy không thể tin một judge đơn lẻ (kappa V1 chỉ 0.32 → V2 0.50). Về kỹ thuật, một hệ eval thật còn phải **robust** trước giới hạn hạ tầng (quota, rate limit, model yếu) — caching, fallback và fail-fast là yếu tố sống còn để pipeline luôn cho ra kết quả.
