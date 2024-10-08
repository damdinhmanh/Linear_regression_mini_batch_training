import numpy as np
import matplotlib.pyplot as plt
import time

def get_column(data, index):
    result = [row[index] for row in data]
    return result

data = np.genfromtxt('data.csv', delimiter=',',)

x_data = get_column(data, 0)
y_data = get_column(data, 1)

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

samples_num = len(x_data)

print(samples_num)

#init weights
b = 0.04
w = -0.34
lr = 0.02

epoch_max = 50
losses_debug = []
plt.ion()

for epoch_i in range(epoch_max):
    for sample_i in range(samples_num):
        #draw current line
        x_draw = range(2, 8)
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


        x = x_data[sample_i]
        y = y_data[sample_i]
        print("current weight: w={:.2f} - b={:.2f}".format( w, b))
        y_hat = predict(x, w, b)

        loss = (y_hat - y) * (y_hat - y)
        losses_debug.append(loss)
        print("loss: {:.2f}\n".format(loss))

        (dw, db) = gradient(y_hat, y, x)
        (w, b) = update_weights(w, b, lr, dw, db)

# plt.plot(losses_debug[10:])
# plt.show()
# plt.pause(10)