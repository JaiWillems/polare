# Polare

Polare is a computation tool that provides a continuous representation of input data, referred to as a Stroke, which doubles as an algebraic node for computations. A set of discrete data can be stored in the Stroke object and operated on as if it was a continuous function; the altered Stroke can then be evaluated within its input domain.

## Why is Polare Useful?

In computation workflows, data can be interpolated after initial processing. Although, in cases where the post-transformation data is vastly more irregular than input data, it can be challenging to capture the actual data behaviour through interpolating the transformed data. In such cases, interpolating the input data before applying the transformations is desirable to capture the behaviour of the underlying data. By design, Strokes take advantage of pre-interpolation without the need first to decide where to interpolate.

The ability to decide interpolation points after the fact can be helpful in many applications. For example, in root-finding algorithms, it is necessary to be able to interpolate at any arbitrary point within the input domain without knowing the root of the data previously. 

In short, if interpolation of irregular transformed data is desired, a more robust and representative interpolation can be achieved by performing the conversion using a Stroke rather than post interpolation.

## Design Focus

The design focus of Polare was for seamless integration into NumPy intensive workflows. As a result, Strokes are compatible with most NumPy universal functions. Currently, NumPy functions that depend on a scalar input (e.g. `np.cos`, `np.radians`, etc.) only work on single Stroke objects whereas NumPy functions that necessitate array inputs (e.g. `np.sum`, `np.dot`, etc), are supported. Additionally, the program supports many Python operands for Stroke-to-Stroke and Stroke-to-scalar operations.


## Quick Start Guide

Firstly, we can use a Stroke for interpolation!

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

Strokes can be operated on using standard Python operators, unlike other interpolation objects.

```python
# We can shift the Stroke through addition/subtraction.
f = f + 10

# We can scale the Stroke through multiplication/division.
f = 5 * f

# And then interpolate the final result...
y_new = f(x_new)
```

Strokes are compatible with many NumPy universal functions allowing for simple integration into NumPy intensive workflows.

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

where `x1,y1,z1` and `x2,y2,z2` represent coordinate points in 3D space. We can then form NumPy vectors:

```python
v1 = np.array([x1, y1, z1])
v2 = np.array([x2, y2, z2])
```

Using standard NumPy techniques, we can get a continuous/interpolable representation of expected results. For example, we can get a continuous/interpolable representation of the norm of our vectors with time:

```python
norm1 = np.linalg.norm(v1)
norm2 = np.linalge.norm(v2)

# Interpolate norms.
t_new = np.linspace(5, 6, 100)
norm1_new = norm1(t_new)
norm2_new = norm2(t_new)
```

Or we could get a continuous/interpolable representation of the vector dot product with time:

```python
dot12 = np.dot(v1, v2)

# Interpolate inner product.
t_new = np.linspace(5, 6, 100)
dot12_new = dot12(t_new)
```

Combining these results, we can get a continuous/interpolable time-series of the angle between our two vectors:

```python
theta12_rad = np.arccos(dot12 / (norm1 * norm2))
theta12_deg = np.degrees(theta12_rad)

# Interpolate angles.
t_new = np.linspace(5, 6, 100)
theta12_rad_new = theta12_rad(t_new)
theta12_deg_new = theta12_deg(t_new)
```

An example of another helpful data operation is matrix-vector multiplication. To rotate our vector, `v1`, about the z-axis by a time-evolving angle, `theta12_rad`, we can use the 3rd principal axis rotation:

```python

# Construct rotation matrix and rotate position vector.
c3 = np.array([[np.cos(theta12_rad), -np.sin(theta12_rad), 0],
               [np.sin(theta12_rad), np.cos(theta12_rad),  0],
               [0,                   0,                    1]])
new_pos = np.matmul(c3, v1)

# Interpolate new (rotated) positions.
t_new = np.linspace(0, 9, 100)
x1_new = new_pos[0](t_new)
y1_new = new_pos[1](t_new)
z1_new = new_pos[2](t_new)
```
