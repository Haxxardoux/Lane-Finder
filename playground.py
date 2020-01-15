import numpy as np 
import random
import matplotlib.pyplot as plt
import time

points = np.arange(0,1000)


means = []
for i in points:
    means.append(random.sample(list(points),k=2))

array = np.array(means).T

print(array[1][0:12])

timedelta = []
for i in range(1,300):
    print(i)
    t0 = time.time()
    fit = np.polyfit(array[0][0:i], array[1][0:i],2)
    t1 = time.time()
    timedelta.append(t1-t0)

plt.plot(timedelta)
plt.show()