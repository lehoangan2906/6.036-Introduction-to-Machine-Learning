# Numpy intro

'''
numpy is a package for doing a variety of numerical computations in Python. We will use it extensively. 
It supports writing very compact and efficient code for handling arrays of data. We will start every code 
file that uses numpy with import numpy as np, so that we can reference numpy functions with the 'np.' precedent.

You can find general documentation on numpy here, and we also have a 6.036-specific numpy tutorial.

The fundamental data type in numpy is the multidimensional array, and arrays are usually generated from 
a nested list of values using the np.array command. Every array has a shape attribute which is a tuple of 
dimension sizes.

In this class, we will use two-dimensional arrays almost exclusively. That is, we will use 2D arrays to represent 
both matrices and vectors! This is one of several times where we will seem to be unnecessarily fussy about how we 
construct and manipulate vectors and matrices, but we have our reasons. We have found that using this format results 
in predictable results from numpy operations.

Using 2D arrays for matrices is clear enough, but what about column and row vectors? We will represent a column vector 
as a d×1 array and a row vector as a 1×d array.So for example, we will represent the three-element column vector,

        |1|
    x = |5|,
        |3|

as a 3×1 numpy array. This array can be generated with
   
    x = np.array([[1],[5],[3]]),

or by using the transpose of a 1×3 array (a row vector) as in,

    x = np.transpose(np.array([[1,5,3]]),

where you should take note of the "double" brackets.

It is often more convenient to use the array attribute .T , as in

    x = np.array([[1,5,3]]).T

to compute the transpose.

Before you begin, we would like to note that in this assignment we will not accept answers that use "loops". One reason for avoiding loops is efficiency. For many operations, numpy calls a compiled library written in C, and the library is far faster than that interpreted Python (in part due to the low-level nature of C, optimizations like vectorization, and in some cases, parallelization). But the more important reason for avoiding loops is that using higher-level constructs leads to simpler code that is easier to debug. So, we expect that you should be able to transform loop operations into equivalent operations on numpy arrays, and we will practice this in this assignment.

Of course, there will be more complex algorithms that require loops, but when manipulating matrices you should always look for a solution without loops.

Numpy functions and features you should be familiar with for this assignment:

- np.array
- np.transpose (and the equivalent method a.T)
- np.ndarray.shape
- np.dot (and the equivalent method a.dot(b) )
- np.sign
- np.sum (look at the axis and keepdims arguments)
- Elementwise operators +, -, *, /

'''

# 1.1 Array
'''
Provide an expression that sets A to be a 2 x 3 numpy array (2 rows by 3 columns),
containing any values you wish.
'''

import numpy as np
A = np.array([[2, 3, 4], [3, 4, 5]])



# 2.2 Transpose
'''
Write a procedure that takes an array and returns the transpose of the array. You can use 
'np.transpose' or the '.T', but you may not use a loop.
'''

import numpy as np
def tp(A):
    return np.array(A).T          



# 2.3 shapes
'''
Let A be a 4 x 2 numpy array, B be a 4 x 3 array, and C be a 4 x 1 array. For each of the 
following expressions, indicate the shape of the result as a tuple of integers (recall python
tuples use parentheses, not square breackets, which are for lisrs, and a tuple of a single
object x is written as (x, ) with a comma) or "none" (as Python string with quotes) if it is 
illegal.
'''

    # 2.3a
        C * C = (4, 1)

    # 2.3b
        np.dot(C, C) = "none"

    # 2.3c
        np.dot(np.transpose(C), C) = (1, 1)

    # 2.3d
        np.dot(A, B) = "none"
    
    # 2.3e
        np.dot(A.T, B)

    #2.3f
        D = np,array([1, 2, 3])
    
    # 2.3g
        A[:,1] = (4, )
    
    # 2.3h
        A[:,1:2] = (4, 1)
    


# 2.4 Row vector
'''
Write a procedure that takes a list of numbers and returns a 2D array representing a 
row vector containing those numbers.
'''

import numpy as np 
def rv(value_list):
    return np.array([value_list])



# 2.5 Column vector
'''
Write a procedure that takes a list of numbers and returns a 2D numpy array representing a 
column vector containing those numbers. You can use the rv procedure.
'''
import numpy as np 
def cv(value_list):
    return np.array([value_list]).T



# 2.6 Length
'''
Write a procedure that takes a column vector and returns the vector's Euclidean length (or
equivalently, its magnitude) as a scalar. You may not use np.linalg.norm, and you may not 
use a loop.
'''

import numpy as np         
def length(col_v):
    return(np.sqrt(snp.sum(col_v**2)))



# 2.7 Normalize
'''
Write a procedure that takes a column vector and returns a unit vector in the same direction. 
You may not use a for loop. Use your length procedure form above (you do not need to 
define it again).
'''

import numpy as np
def normalize(col_v):
    return col_v/length(col_v)



# 2.8 Indexing
'''
Write a procedure that takes a 2D array and returns the final column as a two dimensional
array. You may not use a for loop/
'''

import numpy as np 
def index_final_col(A):
    return A[:,-1:]




# 2.9 Representing data
'''
Alice has collected wieght and height data of 3 people and has written it down below:

Weight,height
150, 5.8
130, 5.5
120, 5.3

She wants to put this into a numpy array such that each row represents one individual's
height and weight in the order listed. Write code to set data equal to the appropriate numpy
array:
'''

import numpy as np
data = np.array([[150, 5.8], [130, 5.5], [120, 5.3]])



'''
Now she wants to compute the sum of each person's height and weight as a column vector
by multiplying data by another numpy array. She has written the following incorrect code to 
do so and needs your helps to fix it:
'''

# 2.10 Matrix multiplication
import numpy as np 
def transform(data):
    return (np.dot(data, np,array([[1], [1]])))
    
