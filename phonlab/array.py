import numpy as np

def nonzero_groups(a, minlen=1, include_any=None):
    '''Return groups of indexes where elements are non-zero. This function
is similar to numpy's `nonzero()`, but instead of returning a single array of
indexes, a tuple of consecutive groups of indexes is returned. Unlike
`nonzero()` only 1d arrays are supported.

Parameters
----------

a : array
The input array to search. The `dtype` must be boolean or coerceable to
boolean.

minlen : int
The minimum length of a run in order to be included in the output.
Runs with fewer than `minlen` elements are excluded.

include_any : scalar or array_like, optional
One or more indexes that the output runs must include. Any runs that
contain at least one of the indexes provided by `include_any` are
in the output. Runs that do not contain at least one of the indexes
are excluded. This parameter does not guarantee that all

Returns
-------

runs : tuple of arrays
Tuple of arrays in which the contents of each array are the indexes
into `a` that represent a run of consecutive `True` values.

'''
    # Coerce to array of (0, 1) values.
    a = np.asanyarray(a).astype(bool).astype(int)

    # By taking the diff of `a` we detect transitions between True
    # and False values. A value of `1` marks a transition from False
    # to True, `-1` marks a transition from True to False, and `0`
    # indicates a steady state from the previous True or False value.
    # `0` is padded to the edges of `a` so that we can detect
    # transitions to/from True values at the beginning and end.
    try:
        states = np.diff(a, prepend=0, append=0)
    except TypeError:  # For older numpy
        states = np.diff(np.hstack([0, a, 0]))

    # Indexes of transitions to true and false.
    true_starts = (states == 1).nonzero()[0]
    false_starts = (states == -1).nonzero()[0]
    try:
        assert(np.all(true_starts < false_starts))
    except:
        raise RuntimeError('Unexpected state in nonzero_groups().')
    if include_any is not None:
        runs = tuple((
            np.arange(tidx, fidx) \
                for tidx, fidx in zip(true_starts, false_starts) \
                    if (fidx - tidx) >= minlen \
                        and np.any(np.isin(np.arange(tidx, fidx), include_any)) 
        ))
    else:
        runs = tuple((
            np.arange(tidx, fidx) \
                for tidx, fidx in zip(true_starts, false_starts) \
                    if (fidx - tidx) >= minlen
        ))
    return runs

def roll_and_pad(a, shift, val):
    '''
Roll an array 'a' by amount 'shift' in a way similar to np.roll(), but
instead of wrapping values around the edges, replace missing edge elements
with constant 'val'.

The intended use case for this function is for rolling a Pandas Series filled
with strings. It will likely work for sequences of other types but might not
work with n-dimensional arrays.

Parameters
----------

a : array_like
    Input array.

shift: int
    The number of places by which elements are shifted.

val : scalar
    The value to set the padded values.

Returns
-------

rolled : array
    The rolled and padded values, as a numpy array.

Examples
--------

>>> roll_and_pad(['a', 'b', 'c'], 1, 'sp')
array(['sp', 'a', 'b'], dtype=object)

>>> roll_and_pad(['a', 'b', 'c'], -2, '')
array(['c', '', ''], dtype=object)

'''
    if shift >= 0:
        index = np.arange(len(a))
    else:
        index = np.arange(len(a) * -1, 0, 1)
# Note the cast to pd.Series for better string handling. Observe the treatment
# of the string as list-like rather than as a unitary value if a is np array:

# In[1]: np.pad(np.array(['a', 'b', 'c']), 2, 'constant', constant_values='XY')
# Out[1]: array(['X', 'X', 'a', 'b', 'c', 'X', 'X'], dtype='<U1')
#
# In [2]: np.pad(pd.Series(['a', 'b', 'c']), 2, 'constant', constant_values='XY')
# Out[2]: array(['XY', 'XY', 'a', 'b', 'c', 'XY', 'XY'], dtype=object)

    if isinstance(val, str):
        a = pd.Series(a)
    return np.pad(a, np.abs(shift), 'constant', constant_values=val)[index]
