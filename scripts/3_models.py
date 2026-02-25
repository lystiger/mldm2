import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from itertools import cycle

# ==========================================
# 1. KHAI BÁO MÔ HÌNH (CẬU CHỈ CẦN THAY ĐỔI Ở ĐÂY)
# ==========================================
# Mở comment dòng cậu muốn chạy, và đóng comment các dòng khác lại:

# --- Lựa chọn 1: SVM (Support Vector Machine) ---
MODEL_NAME = "SVM (RBF Kernel)"
from sklearn.svm import SVC
model = SVC(kernel='rbf', probability=True, random_state=42) # Bắt buộc probability=True để vẽ ROC

# --- Lựa chọn 2: K-Nearest Neighbors (KNN) ---
# MODEL_NAME = "KNN (K=5)"
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=5)

# --- Lựa chọn 3: Logistic Regression ---
# MODEL_NAME = "Logistic Regression"
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=2000, multi_class='ovr', random_state=42)

print(f"⏳ Đang huấn luyện và phân tích chuyên sâu mô hình: {MODEL_NAME}...")

# ==========================================
# 2. CHUẨN BỊ VÀ CHUẨN HÓA DỮ LIỆU
# ==========================================
# (Giả sử X_ML và y_labels là data thật của cậu)
np.random.seed(42)
X_ML = np.random.rand(600, 34) 
y_labels = np.array(['A']*200 + ['B']*200 + ['C']*200)

classes = np.unique(y_labels)
y_bin = label_binarize(y_labels, classes=classes)
n_classes = y_bin.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_ML, y_labels, test_size=0.2, random_state=42)
_, _, y_train_bin, y_test_bin = train_test_split(X_ML, y_bin, test_size=0.2, random_state=42)

# BƯỚC QUAN TRỌNG: FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Nhớ là test data chỉ transform, không fit nhé!

# ==========================================
# 3. HUẤN LUYỆN VÀ DỰ ĐOÁN
# ==========================================
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_score = model.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"🎯 Độ chính xác (Accuracy): {acc * 100:.2f}%\n")

# ==========================================
# 4. VẼ BIỂU ĐỒ (CONFUSION MATRIX & ROC)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Biểu đồ 1: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges' if MODEL_NAME=="SVM (RBF Kernel)" else 'Greens', 
            ax=axes[0], xticklabels=classes, yticklabels=classes)
axes[0].set_title(f'Confusion Matrix - {MODEL_NAME}')
axes[0].set_xlabel('Dự đoán')
axes[0].set_ylabel('Thực tế')

# --- Biểu đồ 2: ROC Curve ---
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color=color, lw=2, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

axes[1].plot([0, 1], [0, 1], 'k--', lw=2)
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_title(f'ROC Curve - {MODEL_NAME}')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

# ==========================================
# 5. XUẤT FILE MODEL (NẾU CHỌN MODEL NÀY ĐỂ DEMO REAL-TIME)
# ==========================================
# LƯU Ý: Nếu dùng SVM/KNN/Logistic để chạy Real-time, bắt buộc phải lưu cả cái SCALER
# joblib.dump(model, 'best_model.pkl')
# joblib.dump(scaler, 'my_scaler.pkl')
