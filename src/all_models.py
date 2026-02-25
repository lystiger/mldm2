import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import 4 "vũ khí" hạng nặng của ML truyền thống
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# ==========================================
# 0. CHUẨN BỊ DỮ LIỆU 
# ==========================================
# Giả lập data (Thay bằng X_ML và y_labels thực tế của cậu)
np.random.seed(42)
X_ML = np.random.rand(600, 34) 
y_labels = np.array(['A']*200 + ['B']*200 + ['C']*200)

X_train, X_test, y_train, y_test = train_test_split(X_ML, y_labels, test_size=0.2, random_state=42)

# ==========================================
# 1. BƯỚC SỐNG CÒN: CHUẨN HÓA DỮ LIỆU (FEATURE SCALING)
# ==========================================
scaler = StandardScaler()
# Cho Scaler học từ tập Train, sau đó biến đổi cả Train và Test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. KHAI BÁO CÁC MÔ HÌNH
# ==========================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42)
}

# ==========================================
# 3. TRAIN VÀ ĐÁNH GIÁ ĐỒNG LOẠT
# ==========================================
print("🚀 BẮT ĐẦU CHẠY ĐUA CÁC MÔ HÌNH...\n")
results = {}

for name, model in models.items():
    # Nhớ dùng data đã Scaled nhé!
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc * 100
    print(f"✅ {name:20s} : Độ chính xác {results[name]:.2f}%")

# ==========================================
# 4. VẼ BIỂU ĐỒ SO SÁNH DÁN VÀO BÁO CÁO
# ==========================================
plt.figure(figsize=(10, 6))
# Vẽ biểu đồ cột
bars = plt.bar(results.keys(), results.values(), color=['#4CAF50', '#2196F3', '#FFC107', '#E91E63'])

# Gắn số % lên đầu mỗi cột cho chuyên nghiệp
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 110) # Để trục Y cao hơn 100 một xíu cho đẹp
plt.title('So sánh Hiệu năng các thuật toán Machine Learning', fontsize=14, fontweight='bold')
plt.ylabel('Độ chính xác (Accuracy %)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()  
