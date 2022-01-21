# Stroke

A stroke is a kernel for continuous data operations that stores a continuous representation of input data. It takes in an `x` and `y` array representing the function `y=f(x)` for scalar `x` and `y`; the object can then be operated on as if it was a point value instead of an array. Practically, a stroke represents
the continuous form of the underlying data allowing it to be evaluated at any time within its input domain.

Currently, the program supports the following scalar-to-stroke and stroke-to-stroke operations in addition to compatibility with NumPy's universal functions:

- Unary positive,
- Unary negative,
- Addition,
- Subtraction,
- Multiplication,
- True division,
- Power, and
- Absolute value.

## When to use a Stroke

The stroke was designed to allow for continuous representation of discrete data operations. In some cases (such as coordinate conversions), input data may be smooth, whereas output data can be very irregular. If interpolation on the output is desired, capturing the actual data behaviour can be challenging. A more robust and accurate fit of irregular data can be achieved by performing the conversion using a stroke with its "continuous" representation.

## Quick Start Guide

Firstly, the stroke can be used for interpolation!

```python
from stroke import Stroke
import numpy as np

# Generate data.
x = np.linspace(-1, 1, 10)
y = np.exp(x) + np.cos(np.pi * x) + 1

# Instantiate Stroke object.
f = Stroke(x=x, y=y, kind="cubic", method="poly")

# Interpolate.
xnew = np.linspace(-1, 1, 100)
ynew = f(xnew)
```

...but with a unique twist. Stroke objects can be operated on using standard
Python operators.

```python
# Generate data.
x = np.linspace(-1, 1, 10)
y = np.full((10,), 1)

# Instantiate Stroke object.
f = Stroke(x=x, y=y, kind="cubic", method="poly")

# We can shift the graph through addition/subtraction.
f = f + 10

# We can scale the graph through multiplication/division.
f = 5 * f

# And then interpolate the final result...
xnew = np.linspace(-1, 1, 100)
ynew = f(xnew)
```

The stroke is also compatible with NumPy's universal functions.

For example, given the time series of cartesian coordinates of a body and a time series of angles, the position components can be rotated about the z-axis by the given angles as follows:

```python
# Generate data.
theta = np.linspace(0, np.pi, 10)
t = np.linspace(0, 9, 10)
x = np.cos(t)
y = t ** 4 - 3
z = 2 * t

# Define Stroke objects and position vector.
ftheta = Stroke(t, theta, kind="cubic")

fx = Stroke(t, x, kind="cubic")
fy = Stroke(t, y, kind="cubic")
fz = Stroke(t, z, kind="cubic")

pos = np.array([fx, fy, fz], dtype=object)

# Construct rotation matrix and rotate position vector.
A11 = np.cos(ftheta)
A12 = -np.sin(ftheta)
A21 = np.sin(ftheta)
A22 = np.cos(ftheta)

M = np.array([[A11, A12, 0], [A21, A22, 0], [0, 0, 1]], dtype=object)
new_pos = np.matmul(M, pos)

# Interpolate new (rotated) positions.
tnew = np.linspace(-1, 1, 100)
xnew = new_pos[0](tnew)
ynew = new_pos[1](tnew)
znew = new_pos[2](tnew)
```
