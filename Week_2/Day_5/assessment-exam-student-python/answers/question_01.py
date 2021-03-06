"""
Create a function that returns the mean of all digits.

Example:
    mean(42) ➞ 3.0

    mean(12345) ➞ 3.0

    mean(666) ➞ 6.0

Notes:
    - Function should always return float
"""


def mean(digits):
    digit = str(digits)
    l = len(digit)
    sum = 0
    for x in digit:
        sum += int(x)
    return(sum/l)