## Mục đích
(Một-hai câu. Liên kết đặc tả thí nghiệm / issue / mục D-xx nếu có.)

## Loại thay đổi
- [ ] Thư viện / mã lõi   - [ ] Thí nghiệm mới (có đặc tả)   - [ ] Dữ liệu / data card   - [ ] Tài liệu   - [ ] Sửa lỗi

## Danh sách kiểm tra
- [ ] `pytest -q` qua (không cần GPU, không cần dữ liệu)
- [ ] Không có dữ liệu, ảnh, khoá truy cập trong commit (`git diff --stat`; kiểm tra `.gitignore`)
- [ ] Cờ/tham số mới mặc định TẮT hoặc không đổi hành vi cũ
- [ ] Nếu là thí nghiệm: có đặc tả đăng ký trước (`docs/EXPERIMENT_SPEC_TEMPLATE.md`), null, B0, CI ghép cặp
- [ ] Nếu đổi quyết định hoặc kết luận: thêm mục vào `docs/DECISIONS.md`
- [ ] Nếu thêm dataset: có data card và dòng trong `docs/DATA_ACCESS.md` (giấy phép, truy cập)
- [ ] Tài liệu cập nhật (README / docs)

## Cách kiểm tra
(Lệnh đã chạy và kết quả; với thí nghiệm: tên suite, số run, thư mục REPORT.)

## Rủi ro / ghi chú cho người review
