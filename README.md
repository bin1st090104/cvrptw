# cvrptw

## Testcases được sử dụng

Gồm 9 test cases có sẵn được dùng nằm ở thư mục `testcases` ở thư mục gốc.

## Các yêu cầu phần mềm cần có

- `python` >= 3.12.
- cài đặt các phụ thuộc trong `requirements.txt`.

## Hướng dẫn chạy các thuật toán LP, IP, CP

Chạy file `milp_runner.py` ở thư mục gốc bằng python như sau:

``` bash
python milp_runner.py
```

Kết quả có sẵn nằm ở trong thư mục `milp_results` ở thư mục gốc.

Tinh chỉnh cấu hính ở trong file `milp_runner.py` ở dòng 394 như sau:

``` python
CURRENT_SOLVER = solve_cvrptw_milp_sat_3d
# loại thuật toán sử dụng, gồm có:
# - solve_cvrptw_milp_cp_sat_2d_optimized
# - solve_cvrptw_milp_sat_3d_with_sequenced_vehicles
# - solve_cvrptw_milp_cp_sat_2d
# - solve_cvrptw_milp_scip_3d
# - solve_cvrptw_milp_scip_3d_with_load_vars
# - solve_cvrptw_milp_sat_3d_with_load_vars
# - solve_cvrptw_milp_sat_2d
# - solve_cvrptw_milp_sat_3d

config = BenchmarkConfig(
    input_folder=Path("testcases"),                           # thư mục chứa test case
    output_dir=Path("milp_results"),                          # thư mục xuất ra kết quả
    output_name="sat_3d_50_100s.txt",                         # Đặt tên kết quả xuất ra
    img_output_dir=Path("milp_results/plots_sat_3d_50_100s"), # thư mục chứa ảnh minh họa kết quả
    visualize=True,                                           # có xuất ra ảnh minh họa hay không
    limit_nodes=50,                                           # giới hạn số khách hàng
    time_limit_sec=100,                                       # giới hạn thời gian
    verbose=True                                              # in ra truy vết kết quả
)
```
