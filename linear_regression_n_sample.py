import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd

def get_column(data, index):
    result = [row[index] for row in data]
    return result

# data = np.genfromtxt('data.csv', delimiter=',',)
#
# x_data = get_column(data, 0)
# y_data = get_column(data, 1)

data = pandas.read_csv('data_pandas.csv')
print(data)
x_data = data[data.columns[0]]
y_data = data[data.columns[1]]


#forward
def predict(x, w, b):
    return x*w + b

#compute gradient
def gradient(y_hat, y, x):
    dw = 2*x*(y_hat - y)
    db = 2*(y_hat - y)
    return (dw, db)

#update weights
def update_weights(w, b, lr, dw, db):
    w_new = w - lr*dw
    b_new = b - lr*db
    return (w_new, b_new)


#init weights
b = 0.04
w = -0.34
lr = 0.02

samples_num = len(y_data)
print('sample num:{}'.format(samples_num))

epoch_max = 10
losses_debug = []
plt.ion()

for epoch_i in range(epoch_max):
    dw_total = 0
    db_total = 0
    loss_total = 0.0

    # draw current epoch_i line
    x_draw = range(2, 10)
    y_draw = [x * w + b for x in x_draw]
    plt.plot(x_draw, y_draw, 'r')
    plt.scatter(x_data, y_data)
    plt.title('Training epoch: '+ str(epoch_i))
    plt.xlabel('Areas')
    plt.ylabel('Prices')
    plt.xlim(3, 7)
    plt.ylim(4, 10)
    plt.show()
    plt.pause(1)
    plt.clf()

    for sample_i in range(samples_num):
        x = x_data[sample_i]
        y = y_data[sample_i]
        print("current weight: w={:.2f} - b={:.2f}".format( w, b))
        y_hat = predict(x, w, b)

        loss = (y_hat - y) * (y_hat - y)
        loss_total = loss_total + loss
        print("loss: {:.2f}\n".format(loss))

        (dw, db) = gradient(y_hat, y, x)
        dw_total = dw_total + dw
        db_total = db_total + db

    losses_debug.append(loss_total/samples_num)
    (w, b) = update_weights(w, b, lr, dw_total/samples_num, db_total/samples_num)

plt.plot(losses_debug)
plt.title('Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.show()
plt.pause(10)