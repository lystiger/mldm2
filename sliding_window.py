import pandas as pd
import numpy as np

# --- CẤU HÌNH CỬA SỔ ---
WINDOW_SIZE = 50  # Số dòng trong 1 cửa sổ (Ví dụ: 50 dòng * 20ms = 1 giây dữ liệu)
STEP_SIZE = 25    # Bước trượt (25 dòng = Overlap 50%. Cửa sổ sau đè lên nửa cửa sổ trước)

def sliding_window(df, window_size, step_size):
    """
    Cắt dataframe thành các khối 3D (Số lượng Window, Window Size, Số lượng Cảm biến)
    """
    windows = []
    labels = []
    
    # Lấy danh sách các nhãn duy nhất (VD: 'A', 'B', 'C')
    unique_labels = df['label'].unique()
    
    for label in unique_labels:
        # Lọc data theo từng hành động để tránh cắt dính râu ông nọ cắm cằm bà kia
        df_label = df[df['label'] == label].copy()
        
        # Bỏ cột label và timestamp đi, chỉ giữ lại 11 cột số (6 MPU + 5 Flex)
        sensor_data = df_label.drop(columns=['label', 'timestamp_ms']).values
        
        # Bắt đầu trượt cửa sổ
        for i in range(0, len(sensor_data) - window_size + 1, step_size):
            window = sensor_data[i : i + window_size]
            windows.append(window)
            labels.append(label)
            
    # Chuyển thành ma trận Numpy (Rất quan trọng cho Machine Learning)
    return np.array(windows), np.array(labels)

# ==========================================
# TEST CHẠY THỬ
# ==========================================
if __name__ == "__main__":
    # 1. Đọc file CSV cậu vừa thu thập được
    try:
        df = pd.read_csv('glove_dataset.csv')
    except FileNotFoundError:
        print("Chưa có file data. Hãy tạo dummy data để test...")
        # Tạo data giả để chạy thử script
        dummy_data = np.random.rand(200, 11) # 200 dòng, 11 cảm biến
        df = pd.DataFrame(dummy_data, columns=['ax','ay','az','gx','gy','gz','f1','f2','f3','f4','f5'])
        df['timestamp_ms'] = range(0, 4000, 20)
        df['label'] = ['A']*100 + ['B']*100
    
    print(f"1. Kích thước Data gốc: {df.shape}")
    
    # 2. Áp dụng Sliding Window
    X_windows, y_labels = sliding_window(df, WINDOW_SIZE, STEP_SIZE)
    
    print(f"2. Kích thước X sau khi Windowing: {X_windows.shape}")
    # Kết quả kỳ vọng: (Số_lượng_window, 50, 11) 
    # Ví dụ: (6, 50, 11) -> 6 mẫu dữ liệu, mỗi mẫu dài 50 frames, có 11 tính năng
    
    print(f"3. Số lượng nhãn y tương ứng: {y_labels.shape}")
