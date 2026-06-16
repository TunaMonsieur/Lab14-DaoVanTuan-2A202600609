# Sổ tay Bảo mật & Quyền riêng tư - NovaCloud

## Mã hóa dữ liệu
Toàn bộ dữ liệu được mã hóa khi lưu trữ (AES-256) và khi truyền tải (TLS 1.3). Khóa mã hóa được quản lý qua hệ thống KMS và xoay vòng định kỳ mỗi 90 ngày. Khách hàng Enterprise có thể dùng khóa tự quản lý (BYOK).

## Phân quyền truy cập
NovaCloud dùng mô hình RBAC với 4 vai trò: Owner, Admin, Editor và Viewer. Chỉ Owner có thể xóa dự án hoặc chuyển quyền sở hữu. Admin quản lý thành viên nhưng không thể xóa Owner. Viewer chỉ có quyền đọc.

## Nhật ký kiểm toán (Audit Log)
Mọi hành động quan trọng (đăng nhập, đổi quyền, xóa dữ liệu) được ghi vào audit log và lưu trong 365 ngày với gói Enterprise, 90 ngày với gói Pro. Gói Free không có audit log. Log không thể bị chỉnh sửa hay xóa thủ công.

## Tuân thủ và chứng chỉ
NovaCloud tuân thủ GDPR, SOC 2 Type II và ISO 27001. Dữ liệu khách hàng EU được lưu tại trung tâm dữ liệu Frankfurt. NovaCloud không bán dữ liệu người dùng cho bên thứ ba.

## Báo cáo lỗ hổng bảo mật
Người dùng báo cáo lỗ hổng qua security@nova.cloud hoặc chương trình bug bounty. NovaCloud cam kết phản hồi trong 48 giờ và không truy cứu pháp lý với nghiên cứu bảo mật có thiện chí (safe harbor).
