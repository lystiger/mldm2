import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ==========================================
# 0. CHUẨN BỊ DỮ LIỆU (Giả lập data nếu cậu chưa có file thật)
# ==========================================
# Trong thực tế, X_ML và y_labels là kết quả trả về từ hàm Feature Engineering lúc nãy
# Ở đây tớ tạo data giả để code có thể chạy ngay lập tức
np.random.seed(42)
X_ML = np.random.rand(600, 34) # 600 mẫu, 34 features
y_labels = np.array(['A']*200 + ['B']*200 + ['C']*200)

# Chuyển đổi nhãn dạng chữ (A, B, C) thành số để tính ROC
classes = np.unique(y_labels)
y_bin = label_binarize(y_labels, classes=classes)
n_classes = y_bin.shape[1]

# Chia tập Train (80%) và Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_ML, y_labels, test_size=0.2, random_state=42)
_, _, y_train_bin, y_test_bin = train_test_split(X_ML, y_bin, test_size=0.2, random_state=42)

# ==========================================
# 1. TRAIN MÔ HÌNH RANDOM FOREST
# ==========================================
print("⏳ Đang huấn luyện mô hình Random Forest...")
# warm_start=True giúp ta vẽ được OOB Error qua từng bước thêm cây
rf_model = RandomForestClassifier(n_estimators=15, warm_start=True, oob_score=True, random_state=42)

# Mảng lưu trữ OOB Error (thay thế cho Loss Curve)
oob_errors = []

# Mô phỏng quá trình thêm dần từng cây vào rừng (từ 15 đến 150 cây)
min_estimators = 15
max_estimators = 150
for i in range(min_estimators, max_estimators + 1):
    rf_model.set_params(n_estimators=i)
    rf_model.fit(X_train, y_train)
    # OOB Error = 1 - OOB Score
    oob_error = 1 - rf_model.oob_score_
    oob_errors.append((i, oob_error))

# Chốt lại model dùng để test
y_pred = rf_model.predict(X_test)
# --- TÌM VÀ IN RA CÁC MẪU BỊ NHẦM LẪN ---
# Lọc ra các vị trí mà Thực tế (y_test) khác với Dự đoán (y_pred)
errors_index = np.where(y_test != y_pred)[0]

print(f"\n🔍 TÌM THẤY {len(errors_index)} MẪU ĐOÁN SAI:")
for idx in errors_index:
    actual = y_test[idx]
    predicted = y_pred[idx]
    
    # Lấy ra bộ đặc trưng của mẫu bị sai đó để soi xét (chỉ in vài thông số chính cho gọn)
    # Ví dụ: Lấy 5 giá trị Flex trung bình (nằm ở 5 cột đầu tiên của X_test)
    flex_vals = X_test[idx][:5] 
    
    print(f"- Lẽ ra là chữ [{actual}] nhưng Model lại đoán là [{predicted}].")
    print(f"  Giá trị Flex sensor lúc đó: {np.round(flex_vals, 2)}")
y_score = rf_model.predict_proba(X_test) # Xác suất cho ROC

# ==========================================
# 2. IN ACCURACY
# ==========================================
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ HUẤN LUYỆN XONG!")
print(f"🎯 Độ chính xác (Accuracy) trên tập Test: {acc * 100:.2f}%")

# ==========================================
# 3. VẼ BIỂU ĐỒ TỔNG HỢP (1 HÌNH CHỨA 3 BIỂU ĐỒ)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- Biểu đồ 1: OOB Error Curve (Thay cho Loss) ---

n_trees, errors = zip(*oob_errors)
axes[0].plot(n_trees, errors, color='red', linewidth=2)
axes[0].set_title('OOB Error Rate (Learning Curve)')
axes[0].set_xlabel('Số lượng Decision Trees')
axes[0].set_ylabel('Tỉ lệ lỗi (OOB Error)')
axes[0].grid(True, linestyle='--', alpha=0.7)

# --- Biểu đồ 2: Confusion Matrix ---

cm = confusion_matrix(y_test, y_pred, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
            xticklabels=classes, yticklabels=classes)
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Dự đoán của Model (Predicted)')
axes[1].set_ylabel('Thực tế (Actual)')

# --- Biểu đồ 3: ROC Curve (Multi-class) ---

colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color=color, lw=2,
                 label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

axes[2].plot([0, 1], [0, 1], 'k--', lw=2) # Đường chéo ngẫu nhiên
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_title('ROC Curve (Đường cong Đặc trưng Hoạt động)')
axes[2].set_xlabel('False Positive Rate (FPR)')
axes[2].set_ylabel('True Positive Rate (TPR)')
axes[2].legend(loc="lower right")
axes[2].grid(True, linestyle='--', alpha=0.7)

# Hiển thị tất cả
plt.tight_layout()
plt.show()
