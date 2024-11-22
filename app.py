from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import arff  # Thư viện để đọc tệp ARFF
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Đọc tệp ARFF
with open('iris.arff') as f:
    dataset = arff.load(f)

# Chuyển đổi dữ liệu ARFF thành DataFrame của pandas
data = np.array(dataset['data'])
df = pd.DataFrame(data, columns=['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class'])

# Xử lý dữ liệu, chuyển đổi nhãn thành dạng số
df['class'] = df['class'].astype('category').cat.codes

# Tách đặc trưng (X) và nhãn (y)
X = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
y = df['class']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tạo mô hình cây quyết định    
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Lấy thông tin từ form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Đưa ra dự đoán
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        predicted_class = dataset['attributes'][-1][1][int(prediction[0])]  # Chuyển đổi dự đoán thành tên loài
        
        return render_template('index.html', predicted_class=predicted_class)
    
    return render_template('index.html', predicted_class=None)

if __name__ == '__main__':
    app.run(debug=True)
