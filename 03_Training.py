
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=== 1. DANG DOC VA TIEN XU LY DU LIEU (TU DONG) ===")
df = pd.read_csv("hotel_bookings.csv")
df = df.drop_duplicates()

leakage_cols = ['reservation_status', 'reservation_status_date', 'assigned_room_type']
X = df.drop(columns=['is_canceled'] + leakage_cols)
y = df['is_canceled']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Dang bien doi du lieu train/test...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n=== 2. HUAN LUYEN VA DANH GIA CAC MO HINH ===")

models = {
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Naive Bayes": GaussianNB()
}

results = []

for name, model in models.items():
    print(f"\n--- Đang huấn luyện mô hình: {name} ---")
    model.fit(X_train_processed, y_train)
    
    y_pred = model.predict(X_test_processed)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        "Thuật toán": name,
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1-Score": round(f1 * 100, 2)
    })

results_df = pd.DataFrame(results)
print("\n===== BANG TONG KET KET QUA CAC THUAT TOAN =====")
print(results_df.to_string(index=False))

print("\nDA HOAN THANH GIAI ĐOẠN HUAN LUYEN VA DANH GIA BAN DAU!")