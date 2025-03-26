from sklearn.metrics import mean_absolute_error
import numpy as np

def relative_error(true_values, predicted_values):
    absolute_error = mean_absolute_error(true_values, predicted_values, multioutput='raw_values')
    return absolute_error / np.abs(true_values)

# 示例数据
true_values = np.array([1.0, 2.0, 3.0, 4.0])
predicted_values = np.array([0.9, 2.1, 2.9, 4.2])

# 计算相对误差
errors = relative_error(true_values, predicted_values)
print(errors)
