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


# Đường dẫn tới tệp JSON lưu trữ lịch sử dự đoán
HISTORY_FILE = 'prediction_history.json'

# Khóa để đồng bộ hóa việc ghi vào tệp JSON
lock = Lock()

# Tải mô hình đã huấn luyện
model_path = os.path.join(os.path.dirname(__file__), 'best_rf_model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")
model = joblib.load(model_path)

# Danh sách các cột trong dữ liệu
columns =["Dân số", "Dân số nam", "Dân số nữ", "Số bệnh viện", "Số trạm y tế", "Số học sinh thpt"]

def load_history():
    """Load prediction history from JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    with lock:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

def save_history(history):
    """Save prediction history to JSON file."""
    with lock:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

@app.route('/',methods=['GET', 'POST'])
def home():
    # Lấy tất cả lịch sử dự đoán từ tệp JSON, sắp xếp theo ngày mới nhất
    history = load_history()
    return render_template('index.html', result=None, history=history)


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
            'Dân số': population,
            'Dân số nam': male,
            'Dân số nữ':female,
            'Số bệnh viện':hospital,
            'Số trạm y tế':subhospital,
            'Số học sinh thpt':thpt
        }


        # Chuyển đổi input_data thành DataFrame
        input_df = pd.DataFrame([input_data])

        # Đảm bảo rằng các cột trong input_df theo đúng thứ tự như mô hình đã huấn luyện
        input_df = input_df.reindex(columns=columns, fill_value=0)


        # Dự đoán
        prediction = model.predict(input_df)[0]

        # Chuyển đổi prediction sang float để JSON có thể serialize
        prediction = int(prediction)

        #Tạo bản ghi dự đoán mới
        new_prediction={
            "id": len(load_history()) + 1,
            "population": population,
            "male":male,
            "female":female,
            "thpt":thpt,
            "hospital":hospital,
            "subhospital":subhospital,
            "result": prediction,
            "date": datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M:%S') 
        }

        # Lấy lịch sử hiện tại, thêm dự đoán mới và lưu lại
        history = load_history()
        history.insert(0, new_prediction)  # Thêm vào đầu danh sách để hiển thị mới nhất lên trên
        save_history(history)


        # Tạo kết quả hiển thị
        result = f"Bác sĩ dự đoán: {prediction} người"

    except ValueError as e:
        result = f"Invalid input: {e}"
    except Exception as e:
        result = f"An error occurred: {e}"

    # Lấy lại lịch sử dự đoán sau khi thêm mới
    history = load_history()

    return render_template('index.html', result=result, history=history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Xóa toàn bộ lịch sử dự đoán bằng cách ghi đè tệp JSON với danh sách rỗng
        save_history([])
        flash("Đã xóa toàn bộ lịch sử dự đoán.", 'success')
    except Exception as e:
        flash(f"Đã xảy ra lỗi khi xóa lịch sử: {e}", 'danger')
    
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)

