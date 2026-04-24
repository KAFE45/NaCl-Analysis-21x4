import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor # สามารถเปลี่ยนเป็น XGBoost, LightGBM หรือ Linear Regression ได้
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. กำหนดชื่อคอลัมน์ที่เป็นคำตอบ (Target)
# ==========================================
# **อย่าลืมเปลี่ยนค่าตรงนี้ให้ตรงกับชื่อคอลัมน์ผลลัพธ์ในไฟล์ CSV ของคุณ**
TARGET_COL = 'target_nacl' 

# ==========================================
# 2. โหลดข้อมูล Training และสร้างโมเดลหลัก
# ==========================================
print("Loading Training Data...")
train_path = r"D:\NaCl-Analysis-21X4\Data\Split\train_all.csv"
train_df = pd.read_csv(train_path)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

print("Training Global Model...")
# ใช้ Random Forest เป็นตัวอย่าง (จำลองเสมือน Software หลัก)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Training Completed!\n")

# ==========================================
# 3. เตรียมไฟล์ Testing ของทั้ง 4 อุปกรณ์
# ==========================================
test_files = {
    "Device 01": r"D:\NaCl-Analysis-21X4\Data\Split\model01_test.csv",
    "Device 02": r"D:\NaCl-Analysis-21X4\Data\Split\model02_test.csv",
    "Device 03": r"D:\NaCl-Analysis-21X4\Data\Split\model03_test.csv",
    "Device 04": r"D:\NaCl-Analysis-21X4\Data\Split\model04_test.csv"
}

# สร้าง List เพื่อเก็บผลลัพธ์
metrics_results = []
all_residuals_df = pd.DataFrame()

# ==========================================
# 4. วนลูปทดสอบและคำนวณความคลาดเคลื่อนรายอุปกรณ์
# ==========================================
for device_name, file_path in test_files.items():
    # โหลดข้อมูลเทสต์ของแต่ละเครื่อง
    test_df = pd.read_csv(file_path)
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    # ทำนายผล
    y_pred = model.predict(X_test)
    
    # คำนวณความคลาดเคลื่อน (Error = Actual - Predicted)
    # ถ้าค่าบวก = เครื่องวัดค่าน้อยกว่าความเป็นจริง, ค่าลบ = เครื่องวัดค่ามากกว่าความเป็นจริง
    residuals = y_test - y_pred 
    
    # เก็บข้อมูล Residual เพื่อนำไปพล็อตราฟ
    temp_res_df = pd.DataFrame({
        'Device': device_name,
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': residuals
    })
    all_residuals_df = pd.concat([all_residuals_df, temp_res_df], ignore_index=True)
    
    # คำนวณ Metrics เพื่อประเมินประสิทธิภาพ
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mean_error = np.mean(residuals) # ดู Bias ของเครื่อง (Systematic Error)
    std_error = np.std(residuals)   # ดูความแกว่งของเครื่อง (Random Error/Variance)
    
    metrics_results.append({
        'Device': device_name,
        'RMSE': rmse,
        'MAE': mae,
        'Mean Error (Bias)': mean_error,
        'Error Std Dev (Variance)': std_error
    })

# ==========================================
# 5. สรุปผลลัพธ์และแสดงกราฟวิเคราะห์ Variation
# ==========================================
results_df = pd.DataFrame(metrics_results)
print("=== สรุปผลการทดสอบแยกตามอุปกรณ์ ===")
print(results_df.to_string(index=False))

# พล็อต Boxplot เพื่อดูการกระจายตัวของความคลาดเคลื่อน (Variation Distribution)
plt.figure(figsize=(10, 6))
sns.boxplot(data=all_residuals_df, x='Device', y='Error', palette='Set2')
plt.axhline(0, color='red', linestyle='--', label='Zero Error (Perfect)')
plt.title('Hardware Variation Analysis (Residual/Error Distribution by Device)')
plt.ylabel('Error (Actual - Predicted)')
plt.xlabel('Device Hardware')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()
