# Sổ tay Kỹ thuật & API - NovaCloud

## Giới hạn API (Rate Limit)
API NovaCloud giới hạn 100 request/phút cho gói Free, 1000 request/phút cho gói Pro và tùy chỉnh cho Enterprise. Khi vượt giới hạn, hệ thống trả về mã lỗi HTTP 429 kèm header Retry-After cho biết số giây cần chờ.

## Xác thực API
API dùng Bearer token. Người dùng tạo API key trong Cài đặt > Developer > API Keys. Mỗi key có thể giới hạn phạm vi (scope) và IP cho phép. Key bị lộ nên được thu hồi ngay và tạo key mới; key cũ ngừng hoạt động tức thì.

## Webhook
NovaCloud gửi webhook cho các sự kiện như project.created, billing.failed. Webhook được ký bằng HMAC-SHA256 trong header X-Nova-Signature. Nếu endpoint trả về lỗi, hệ thống thử lại tối đa 5 lần với khoảng cách tăng dần (exponential backoff).

## SDK và thư viện
NovaCloud cung cấp SDK chính thức cho Python, JavaScript và Go. Các SDK cộng đồng cho Ruby và PHP không được hỗ trợ chính thức. Phiên bản API hiện tại là v2; v1 sẽ ngừng hỗ trợ vào ngày 31/12/2026.

## Giới hạn kích thước
Mỗi request API có giới hạn payload 10MB. File upload qua API tối đa 5GB và phải dùng multipart upload cho file trên 100MB. Timeout mặc định của request là 30 giây.
