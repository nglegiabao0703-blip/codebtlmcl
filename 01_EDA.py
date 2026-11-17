
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("hotel_bookings.csv")


print("===== TONG QUAN DU LIEU =====")
print("So dong:", df.shape[0])
print("So cot:", df.shape[1])


print("\n===== 5 DONG DAU TIEN =====")
print(df.head())


print("\n===== THONG TIN CAC COT =====")
df.info()


print("\n===== DU LIEU THIEU =====")
print(df.isnull().sum())


print("\n===== DU LIEU TRUNG LAP =====")
print("So dong trung lap:", df.duplicated().sum())

print("\n===== THONG KE MO TA =====")
print(df.describe())
