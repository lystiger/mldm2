import numpy as np
from ahrs.filters import Madgwick

def extract_features_from_window(window):
    """
    Input: window có shape (50, 11)
    Output: Vector 1D có 34 features
    """
    # 0. Bóc tách dữ liệu
    acc = window[:, 0:3]
    gyro = window[:, 3:6] * (np.pi / 180.0) # Madgwick cần Radian/s
    flex = window[:, 6:11]

    # ==========================================
    # 1. ĐẶC TRƯNG FLEX SENSOR (20 Đặc trưng)
    # ==========================================
    flex_mean = np.mean(flex, axis=0)
    flex_std = np.std(flex, axis=0)
    flex_max = np.max(flex, axis=0)
    flex_min = np.min(flex, axis=0)

    # ==========================================
    # 2. ĐẶC TRƯNG TƯ THẾ (QUATERNION) (4 Đặc trưng)
    # ==========================================
    madgwick = Madgwick()
    Q = np.zeros((len(acc), 4))
    Q[0] = [1.0, 0.0, 0.0, 0.0] 
    for t in range(1, len(acc)):
        Q[t] = madgwick.updateIMU(Q[t-1], gyr=gyro[t], acc=acc[t])
    final_quaternion = Q[-1] # Lấy tư thế chốt hạ ở cuối cửa sổ

    # ==========================================
    # 3. NÂNG CẤP 1: MEAN CỦA ACCEL & GYRO (6 Đặc trưng)
    # ==========================================
    acc_mean = np.mean(acc, axis=0)
    gyro_mean = np.mean(gyro, axis=0)

    # ==========================================
    # 4. NÂNG CẤP 2: ĐỘ LỚN VECTOR (MAGNITUDE) (4 Đặc trưng)
    # ==========================================
    # Tính Magnitude cho từng frame (50 frame -> mảng 50 phần tử)
    # Dùng np.linalg.norm là cách tính sqrt(x^2 + y^2 + z^2) nhanh nhất trong Numpy
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    # Rút trích Mean và Std từ mảng Magnitude đó (nhớ bọc trong list [] để lát gộp cho dễ)
    acc_mag_mean = [np.mean(acc_mag)]
    acc_mag_std = [np.std(acc_mag)]
    
    gyro_mag_mean = [np.mean(gyro_mag)]
    gyro_mag_std = [np.std(gyro_mag)]

    # ==========================================
    # 5. GỘP TOÀN BỘ THÀNH 1 VECTOR DUY NHẤT
    # ==========================================
    # 20 + 4 + 3 + 3 + 1 + 1 + 1 + 1 = 34 Đặc trưng!
    feature_vector = np.concatenate([
        flex_mean, flex_std, flex_max, flex_min,
        final_quaternion,
        acc_mean, gyro_mean,
        acc_mag_mean, acc_mag_std,
        gyro_mag_mean, gyro_mag_std
    ])
    
    return feature_vector

# Test thử xem mảng đầu ra có đúng 34 phần tử không
# dummy_window = np.random.rand(50, 11)
# f_vec = extract_features_from_window(dummy_window)
# print(f"Kích thước vector đặc trưng: {f_vec.shape}") # Sẽ in ra (34,)
