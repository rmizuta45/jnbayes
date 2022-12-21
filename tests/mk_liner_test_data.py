import numpy as np

n = 100
sigma_R = 0.5#ノイズの標準偏差
x = np.arange(n)/10
a = 1.9
b = 5.5

y = a*x + b + np.random.normal(loc=0, scale=sigma_R,size=n)


data = [x,y]

np.save("dt/test/liner.npy",data)

