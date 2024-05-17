import pandas as pd
import numpy as np
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Đọc dữ liệu
    data = pd.read_csv('USA_housing.csv')
    data = data.drop('Address', axis=1)

    # Tách dữ liệu thành các biến đầu vào và biến đầu ra
    X = data.drop('Price', axis=1).values
    Y = data['Price'].values

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (70% cho huấn luyện, 30% cho kiểm tra)
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Thêm cột hệ số chặn vào X_train
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    # Tính toán các hệ số hồi quy beta = (X^T * X)^(-1) * X^T * Y
    XTX = np.dot(X_train.T, X_train)
    XTX_inv = np.linalg.inv(XTX)
    XTY = np.dot(X_train.T, Y_train)
    beta = np.dot(XTX_inv, XTY)

    # Hàm dự đoán giá nhà dựa trên các giá trị đầu vào
    def predict_price(var1, var2, var3, var4, var5):
        input_vars = np.array([1, var1, var2, var3, var4, var5])  # Thêm hệ số chặn
        return np.dot(input_vars, beta)

    # Giả sử bạn nhận các giá trị từ request GET
    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])

    pred = predict_price(var1, var2, var3, var4, var5)
    pred = round(pred)

    price = "Giá dự đoán là:  $" + str(pred)
    return render(request, 'predict.html', {"result2": price})
