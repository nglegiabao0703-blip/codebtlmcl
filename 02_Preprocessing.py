
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("=== 1. DANG DOC DU LIEU ===")
df = pd.read_csv("hotel_bookings.csv")
print(f"Kich thuoc ban dau: {df.shape}")

print("\n=== 2. XU LY DU LIEU TRUNG LAP ===")

initial_rows = len(df)
df = df.drop_duplicates()
print(f"Da xoa {initial_rows - len(df)} dong trung lap. Con lai: {len(df)} dong.")

print("\n=== 3. LOAI BO DATA LEAKAGE VA COT KHONG CAN THIET ===")

leakage_cols = ['reservation_status', 'reservation_status_date', 'assigned_room_type']

X = df.drop(columns=['is_canceled'] + leakage_cols)
y = df['is_canceled']

print(f"Kich thuoc tap dac trung X: {X.shape}")
print(f"Kich thuoc bien muc tieu y: {y.shape}")

print("\n=== 4. PHAN TICH VA XU LY DU LIEU THIEU ===")

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"So luong dac trung so: {len(numeric_features)}")
print(f"So luong dac trung phan loai: {len(categorical_features)}")


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("\n=== 5. CHIA TAPS TRAIN / TEST (80/20) ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Kich thuoc X_train: {X_train.shape}")
print(f"Kich thuoc X_test: {X_test.shape}")

print("\n=== 6. AP DỤNG PIPELINE TIEN XU LY ===")
print("Dang xu ly va bien doi du lieu (qua trinh nay co the mat vai giay)...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Kich thuoc sau khi xu ly (Train): {X_train_processed.shape}")
print(f"Kich thuoc sau khi xu ly (Test): {X_test_processed.shape}")

print("\nDA HOAN THANH TIEN XU LY PIPELINE A!")