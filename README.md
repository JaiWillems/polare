# Polare

Polare defines a stroke object as a kernel for continuous data transformations that stores a "continuous" representation of input data. It takes in an `x` and `y` array representing the function `y=f(x)` for scalar `x` and `y`; the object can then be operated on as if it was a point value instead of an array. Practically, a stroke represents the continuous form of the underlying data allowing it to be evaluated within its input domain. It is best motivated through the examples below.

Currently, the program supports the following scalar-to-stroke and stroke-to-stroke operations in addition to compatibility with NumPy's universal functions:

- Unary positive,
- Unary negative,
- Addition,
- Subtraction,
- Multiplication,
- True division,
- Power, and
- Absolute value.

## When to use Polare

The stroke is designed to allow for a continuous representation of discrete data transformations and is necessary for transformations with smooth input data but irregular output data. If interpolation on the output is desired, capturing the actual data behaviour can be challenging. A more robust and accurate fit of irregular data can be achieved by performing the conversion using a stroke with its "continuous" representation.

## Quick Start Guide

Firstly, the stroke can be used for interpolation!

```python
from polare import Stroke
import numpy as np

# Generate data.
x = np.linspace(-1, 1, 10)
y = np.exp(x) + np.cos(np.pi * x) + 1

# Instantiate Stroke object.
f = Stroke(x=x, y=y, kind="cubic")

# Interpolate.
xnew = np.linspace(-1, 1, 100)
ynew = f(xnew)
```

...but with the twist that Stroke objects can be operated on using standard
Python operators.

```python
# We can shift the stroke through addition/subtraction.
f = f + 10

# We can scale the stroke through multiplication/division.
f = 5 * f

# And then interpolate the final result...
xnew = np.linspace(-1, 1, 100)
ynew = f(xnew)
```

The stroke is also compatible with NumPy's universal functions allowing for
simple integration into NumPy intensive data handling workflows.

For the following NumPy-focused examples, we will use the proceeding variable definitions:

```python
t = np.linspace(0, 10, 100)

x1 = Stroke(t, x1_data, kind="cubic")
y1 = Stroke(t, y1_data, kind="cubic")
z1 = Stroke(t, z1_data, kind="cubic")

x2 = Stroke(t, x1_data, kind="cubic")
y2 = Stroke(t, y1_data, kind="cubic")
z2 = Stroke(t, z1_data, kind="cubic")
```

where `x1,y1,z1` and `x2,y2,z2` represent the coordinates of two points. We can then form NumPy vectors:

```python
v1 = np.array([x1, y1, z1])
v2 = np.array([x2, y2, z2])
```

We can then use standard NumPy techniques to gain interpolable representations of expected results. We can get a continuous and interpolable representation of the norm of our vectors with time:

```python
norm1 = np.linalg.norm(v1)
norm2 = np.linalge.norm(v2)

# Interpolate norms.
tnew = np.linspace(5, 6, 100)
norm1_new = norm1(tnew)
norm2_new = norm2(tnew)
```

Or we could get a continuous and interpolable representation of the dot product between our vectors with time:

```python
dot12 = np.dot(v1, v2)

# Interpolate inner product.
tnew = np.linspace(5, 6, 100)
dot12_new = dot12(tnew)
```

Combining these results, we can get a continuous and interpolable time-series of the angle between our two vectors:

```python
theta12_rad = np.arccos(dot12 / (norm1 * norm2))
theta12_deg = np.degrees(theta12_rad)

# Interpolate angles.
tnew = np.linspace(5, 6, 100)
theta12_rad_new = theta12_rad(tnew)
theta12_deg_new = theta12_deg(tnew)
```

An example of another helpful data operation is matrix-vector multiplication. To rotate our vectors about the z-axis by a time-evolving angle, `theta12_rad`, we can use the 3rd principal axis rotation:

```python

# Construct rotation matrix and rotate position vector.
C3 = np.array([[np.cos(theta12_rad), -np.sin(theta12_rad), 0],
               [np.sin(theta12_rad), np.cos(theta12_rad),  0],
               [0,                   0,                    1]], dtype=object)
new_pos = np.matmul(C3, v1)

# Interpolate new (rotated) positions.
tnew = np.linspace(0, 9, 100)
x1new = new_pos[0](tnew)
y1new = new_pos[1](tnew)
z1new = new_pos[2](tnew)
```
