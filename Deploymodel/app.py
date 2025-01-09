# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import numpy as np
import json
import os
from datetime import datetime, timezone
from threading import Lock

app = Flask(__name__)

# Cấu hình Secret Key cho Flask (cần thiết cho Flash messages)
app.config['SECRET_KEY'] = 'your_secure_secret_key_here'  # Thay thế bằng khóa bí mật của bạn

# Khóa để đồng bộ hóa việc ghi vào tệp JSON
lock = Lock()

# Tải mô hình đã huấn luyện
model_path = os.path.join(os.path.dirname(__file__), 'best_rf_model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")
model = joblib.load(model_path)

# Danh sách các cột trong dữ liệu
columns = ['Population','Male',"Female","Hospital","Subhospital","THPT"]



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        population=int(request.form.get('population', 0))
        male=int(request.form.get('male', 0))
        female=int(request.form.get('female', 0))
        thpt=int(request.form.get('thpt', 0))
        hospital=int(request.form.get('hospital', 0))
        subhospital=int(request.form.get('subhospital', 0))

        # Chuẩn bị dữ liệu đầu vào cho mô hình
        input_data = {
            'Population': population,
            'Male':male,
            'Female':female,
            'THPT':thpt,
            'Hospital':hospital,
            'Subhospital':subhospital
        }


        # Chuyển đổi input_data thành DataFrame
        input_df = pd.DataFrame([input_data])


        # Dự đoán
        prediction = model.predict(input_df)[0]

        # Chuyển đổi prediction sang float để JSON có thể serialize
        prediction = float(prediction)


        # Tạo kết quả hiển thị
        result = f"Bác sĩ dự đoán: {prediction:,.2f} %"

    except ValueError as e:
        result = f"Invalid input: {e}"
    except Exception as e:
        result = f"An error occurred: {e}"


    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)

