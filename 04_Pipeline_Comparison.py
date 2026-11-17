# ==========================================
# DE TAI: DU DOAN KHACH HANG HUY DAT PHONG
# SO SANH PIPELINE A (CO BAN) VA PIPELINE B (CAI TIEN)
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=== 1. DANG DOC DU LIEU THO ===")
df = pd.read_csv("hotel_bookings.csv")
df = df.drop_duplicates()

# Loai bo Data Leakage
leakage_cols = ['reservation_status', 'reservation_status_date', 'assigned_room_type']
df = df.drop(columns=leakage_cols)

# ==========================================
# DINH NGHIA PIPELINE A (CO BAN)
# ==========================================
def run_pipeline_A(data):
    X = data.drop(columns=['is_canceled'])
    y = data['is_canceled']
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    cat_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_trans, num_cols), 
        ('cat', cat_trans, cat_cols)
    ])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), preprocessor

# ==========================================
# DINH NGHIA PIPELINE B (CAI TIEN - FEATURE ENGINEERING)
# ==========================================
def run_pipeline_B(data):
    df_b = data.copy()
    
    # 1. Tao dac trung moi co y nghia (Feature Engineering)
    df_b['total_nights'] = df_b['stays_in_weekend_nights'] + df_b['stays_in_week_nights']
    df_b['total_guests'] = df_b['adults'] + df_b['children'].fillna(0) + df_b['babies']
    
    # 2. Xu ly thieu thong minh: Tao bien chi bao co su dung company/agent hay khong
    df_b['has_company'] = df_b['company'].notnull().astype(int)
    df_b['has_agent'] = df_b['agent'].notnull().astype(int)
    
    X = df_b.drop(columns=['is_canceled'])
    y = df_b['is_canceled']
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    # Pipeline B dung constant 'Unknown' cho bien phan loai bi thieu
    cat_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_trans, num_cols), 
        ('cat', cat_trans, cat_cols)
    ])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), preprocessor

print("\n=== 2. THUC HIEN THU NGHIEM VA SO SANH HAI PIPELINE ===")

# Chay Pipeline A
(X_train_a, X_test_a, y_train_a, y_test_a), prep_a = run_pipeline_A(df)
X_train_a_proc = prep_a.fit_transform(X_train_a)
X_test_a_proc = prep_a.transform(X_test_a)

# Chay Pipeline B
(X_train_b, X_test_b, y_train_b, y_test_b), prep_b = run_pipeline_B(df)
X_train_b_proc = prep_b.fit_transform(X_train_b)
X_test_b_proc = prep_b.transform(X_test_b)

# Huan luyen mo hinh Decision Tree (max_depth=10) tren ca hai pipeline de so sanh
models_test = {
    "Pipeline A (Cơ bản)": (X_train_a_proc, X_test_a_proc, y_train_a, y_test_a),
    "Pipeline B (Cải tiến - Thêm đặc trưng)": (X_train_b_proc, X_test_b_proc, y_train_b, y_test_b)
}

results = []
for name, (X_tr, X_te, y_tr, y_te) in models_test.items():
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    
    results.append({
        "Cấu hình thử nghiệm": name,
        "Accuracy (%)": round(accuracy_score(y_te, y_pred) * 100, 2),
        "Precision (%)": round(precision_score(y_te, y_pred, zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(y_te, y_pred, zero_division=0) * 100, 2),
        "F1-Score (%)": round(f1_score(y_te, y_pred, zero_division=0) * 100, 2)
    })

print("\n===== BANG SO SANH KET QUA GIUA PIPELINE A VA PIPELINE B =====")
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
print("\nHOAN THÀNH THU NGHIEM SO SANH HAI PIPELINE!")