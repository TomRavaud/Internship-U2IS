import numpy as np
import matplotlib.pyplot as plt
import pywt


x = np.arange(-6, 6, 0.01)
y = np.sin(x**2)

plt.subplot(611)
plt.plot(y)

coefficients = pywt.wavedec(y, 'db4', level=4)

print(len(y))

total = 0

for i in range(len(coefficients)):
    print(len(coefficients[i]))
    total += len(coefficients[i])
    
approximation, details = coefficients[0], coefficients[1:]

plt.subplot(612)
plt.plot(approximation)

plt.subplot(613)
plt.plot(details[0])

plt.subplot(614)
plt.plot(details[1])

plt.subplot(615)
plt.plot(details[2])

plt.subplot(616)
plt.plot(details[3])

plt.show()

# scalogram = dwt_scalogram(coefficients[1:])

# print(scalogram.shape)

# plt.subplot(212)
# plt.imshow(scalogram, cmap='viridis', aspect='auto')
# plt.show()
