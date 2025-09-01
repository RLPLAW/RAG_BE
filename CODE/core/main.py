from pipeline.query_pipeline import init_db, answer_query

if __name__ == "__main__":
    try:
        # Load or create DB
        db = init_db()
        if not db:
            raise RuntimeError("Không thể khởi tạo hoặc tải Chroma DB.")

        # Get report name
        report_name = input(
            "Nhập tên file báo cáo (VD: 676-BÁO CÁO Kết quả công tác thi hành án dân sự tháng 4 năm 2025.pdf): "
        ).strip()
        if not report_name:
            raise ValueError("Tên file báo cáo không được để trống.")

        # Get edit instructions
        instruction = input(
            "Bạn muốn chỉnh sửa gì (VD: đổi tháng từ 4/2025 thành 5/2025, giữ nguyên số liệu): "
        ).strip()
        if not instruction:
            raise ValueError("Hướng dẫn chỉnh sửa không được để trống.")

        # Call pipeline
        result = answer_query(db, report_name, instruction)

        # Check result
        if result:
            print("\nModified Report:\n")
            print(result)
        else:
            print("\nKhông tạo được báo cáo chỉnh sửa. Kiểm tra lại tên file hoặc nội dung.")

    except Exception as e:
        print(f"\nLỗi: {e}")
