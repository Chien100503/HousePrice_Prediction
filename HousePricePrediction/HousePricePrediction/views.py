import pandas as pd
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    data = pd.read_csv('USA_housing.csv')
    data = data.drop('Address', axis=1)
    
    X = data.drop('Price', axis=1)
    Y = data['Price']

    # Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra (30% cho kiểm tra)
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    # Tính toán các giá trị trung bình của các thuộc tính
    avg_values = X_train.mean(axis=0)
    
    # Hàm dự đoán giá nhà dựa trên các giá trị trung bình của thuộc tính
    def predict_price(var1, var2, var3, var4, var5):
        return avg_values[0] * var1 + avg_values[1] * var2 + avg_values[2] * var3 + avg_values[3] * var4 + avg_values[4] * var5
    
    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    
    pred = predict_price(var1, var2, var3, var4, var5)
    pred = round(pred)
    
    price = "The predicted price is: $" + str(pred)
    return render(request, 'predict.html', {"result2": price})
