import numpy as np
import matplotlib.pyplot as plt

"""
    This script plots saturation functions used
    in our adjusted SPLADE model.
"""

# Define the range for the input values with logs starting from 1 and square root starting from 0
x = np.linspace(0, 100, 400)

# Calculate the square root for the full range including 0
square_root_full = np.sqrt(x)
# Calculate log base 2 and log base 10 for the range starting from 1
logarithm_base_2_log = np.log2(x + 1)
logarithm= np.log(x + 1)
log_2_sqrt = np.log(1 + square_root_full)
tanh = np.tanh(x)

sqr_inverse = (1 - (1 / (1 + np.square(x)))) * (1 - np.exp(-x / 5))

sigmoid = (1 / (1 + np.exp(-x))) - 0.5

# Plotting the functions
plt.figure(figsize=(10, 5))
plt.plot(x, square_root_full, label='Square Root ($\\sqrt{x}$)')
plt.plot(x, logarithm_base_2_log, label='Logarithm Base 2 ($\\log_2(x)$)')
plt.plot(x, logarithm, label='Natural Logarithm ($\\ln(x)$)')
plt.plot(x, sigmoid, label='Sigmoid Zero Centered ($\\frac{1}{1 + e^{-x}} - \\frac{1}{2}$)')
plt.plot(x, tanh, label='Hyperbolic Tangent ($\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$)')
plt.title('Growth of Saturation Functions')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.legend()
plt.grid(True)
plt.show()