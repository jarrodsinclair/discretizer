Discretizer
===========

A Python library to discretize floating point numbers into compact bytes using
variable sized buckets. Sigmoid and cube-root mapping functions can be used to
efficiently discretize a number range with high resolution about a specific
number, and less resolution where it is not needed. A linear function can be
used for even distributions. Clamping occurs for numbers outside the chosen
range.

Usage in Python
---------------

Using the linear mapping function to discretize some numbers:

```python
from discretizer import LinearDiscretizer

# create a linear discretizer for the range [-10, 20] into 1 byte
d = LinearDiscretizer(1, -10.0, 20.0)

# discretize some numbers into bytearrays
nums = [-11.0, -10.0, -5.0, 0.0, 15.0, 19.0, 20.0, 21.0]
ba_list = []
for n in nums:
    ba_list.append(d.encode(n))

# covert back to numbers and compare
#  (differences exist due to discretization)
for i in range(len(nums)):
    n = d.decode(ba_list[i])
    print('Original = %f, After discretization = %f, Difference = %f' % \
        (nums[i], n, nums[i]-n))
```

Using the sigmoid mapping function to provide greater accuracy about the middle
of the range:

```python
from discretizer import SigmoidDiscretizer

# create a sigmoid discretizer for the range [-5, 5] into 1 byte
#  using a sharpness parameter of 20
d = SigmoidDiscretizer(1, -5.0, 5.0, 20.0)

# discretize some numbers into bytearrays
nums = [-5.0, -4.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 4.0, 5.0]
ba_list = []
for n in nums:
    ba_list.append(d.encode(n))

# covert back to numbers and compare
#  (differences exist due to discretization)
for i in range(len(nums)):
    n = d.decode(ba_list[i])
    print('Original = %f, After discretization = %f, Difference = %f' % \
        (nums[i], n, nums[i]-n))
```
