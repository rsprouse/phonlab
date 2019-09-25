# Utility functions.

import os, sys
import pandas as pd
import numpy as np
import re
import fnmatch
import shutil
import pkg_resources

def phonemap(k, v, type='df'):
    '''Return a mapping between phones in different transcription systems as
a dataframe with each phone set in a column. The rows contain correspondences
between phone sets.

Phone sets available for mapping are Unicode IPA ('ipa'), ARPABET ('arpabet'),
and GlobalPhone ('globalphone').

Parameters
----------

k : str (one of ['arpabet', 'globalphone', 'ipa']
    The name of the phone set in the first column of the dataframe.

v : str (one of ['arpabet', 'globalphone', 'ipa']
    The name of the phone set in the second column of the dataframe.

Returns
-------

d : dataframe or dict
    The dataframe that contains the phone mappings.
'''
    d = pd.read_csv(
        pkg_resources.resource_stream('phonlab', 'data/phonemap.txt'),
        sep='\t'
    )
    d = d.loc[:, (k, v)].dropna().reset_index(drop=True)
    d = d[~d.duplicated()]
    for col1, col2 in ((k, v), (v, k)):
        for name, group in d[d[col1].duplicated(keep=False)].groupby(col1):
            msg = 'WARNING: {:} phone {:} maps to multiple {:} phones [{:}].\n'
            sys.stderr.write(
                msg.format(col1, name, col2, ', '.join(group.loc[:, col2]))
            )
    return d

def cp_backup(fname, bkdir=None, hidden=True):
    '''Make a backup copy of the file `fname` and return the name of the copied
file. By default the copy will have the same name as `fname` with '.' prepended
and a suffix of the form '.N', where N is an integer. Multiple calls to this
function result in increasing values of N, starting with '1'.

Parameters
----------

fname : str
    The name of the file to be copied.

bkdir : str (default: None)
    By default, the backup file will be written in the same directory
    as the source file. If `bkdir` is provided, the backup file
    will be written to that path instead.

    A `FileNotFoundError` will be thrown if the backup directory does not
    exist.

hidden : bool (default: True)
    If True, prepend '.' to the backup filename, resulting in a
    'hidden' file. If not True, do not prepend anything.

Returns
-------

dst : str
    The name of the copied backup file.
'''
    if bkdir is None:
        bkdir = os.path.dirname(fname) if os.path.dirname(fname) != '' else '.'
    basename = os.path.basename(fname)
    cpname = '.' + basename if hidden is True else basename

    # Get extension integers from backups that already exist and find max N.
    rgx = re.compile(
        fnmatch.translate(cpname).replace('\Z', '\.[0-9]+\Z')
    )
    Ns = np.array([
        int(os.path.splitext(f)[1].lstrip('.')) \
            for f in os.listdir(bkdir) if rgx.fullmatch(f)
    ], dtype=int)
    maxN = 0 if len(Ns) == 0 else Ns.max()

    cpname += '.{:d}'.format(maxN + 1)
    dst = os.path.join(bkdir, cpname)
    shutil.copyfile(fname, dst)
    return dst


def dir2df(dirname, addcols=[], fnpat=None, dirpat=None, sentinel='.bad.txt',
to_datetime=True, dotfiles=False, dotdirs=False, **kwargs):
    '''dir2df(): Recursively generate the filenames in a directory tree
using os.walk() and store as rows of a DataFrame.

'Hidden' files and directories (those with names that start with '.') are
ignored by default. dir2df() will also not descend into a directory tree that
contains a sentinel file (default name '.bad.txt').

Additional parameters can be used to filter which filepaths to include in the
output, and also to add additional file metadata.

Parameters
----------

dirname : str
    Top-level directory name for filename search.


Optional parameters
-------------------

fnpat : str, re
    Regular expression pattern that defines the filenames to return.
    The only filenames in the result set will be those that return a match
    for `re.search(fnpat, filename)`.

    If you use named captures in `fnpat`, new columns of dtype 'Categorical'
    will appear in the output that correspond to the capture group contents.

    If you need to use a flag with your pattern, you can use a precompiled
    regex for the value of `fnpat`. For example, you can do
    case-insensitive matching of '.wav' and '.WAV' files with
    `re.compile(r'\.wav$', re.IGNORECASE)`.

dirpat : str, re
    Like `fnpat`, only applied against the relative path in dirname.
    Relative paths that do not match `dirpat` will be skipped.

sentinel : str (default '.bad.txt')
    Name of the sentinel file, which marks a directory tree to be ignored. No
    filenames from the directory containing the sentinel file will be included
    in the output, nor will any filenames from any of its subdirectories.
    If the value of `sentinel` is '' or None, the sentinel file check will not
    be performed.

addcols = str, list of str (default [])
    One or more additional columns to include in the output. Possible names
    and values provided are:
    'dirname': the user-provided top-level directory
    'barename': the filename without path or extension
    'ext': the filename extension
    'mtime': the last modification time of the file
    'bytes': the size of the file in bytes

    The 'mtime' column is cast to Pandas Timestamps automatically unless
    `to_datetime` is False. Resolution of the time-based stats is dependent
    on your platform; see the os.stat() documentation.

to_datetime : boolean (default True)
    If True, 'mtime' stats will be converted from Unix epoch to datetime.
    If False, the values will not be converted.

dotfiles : boolean (default False)
    If True, include filenames beginning with `.` in the output. Otherwise,
    omit these names.

dotdirs : boolean (default False)
    If True, descend into directories with names that begin with `.`. If
    False, do not descend into these directories.

kwargs : various
    Remaining kwargs are passed to os.walk(). If not used, then os.walk() will
    be called with default kwargs. Note that using os.walk(topdown=False) is
    not compatible with `dotdirs=False`.


Returns
-------

fnamedf : DataFrame
    Pandas DataFrame with filenames recorded in rows.

'''
    # Coerce addcols to list if passed as single string.
    try:
        assert isinstance(addcols, basestring) # Python 2
        addcols = [addcols]
    except AssertionError:
        pass   # Should be a list already.
    except NameError:
        try:
            assert isinstance(addcols, str) # Python 3
            addcols = [addcols]
        except AssertionError:
            pass   # Should be a list already.

    if 'dirname' in addcols:
        firstcols = ['dirname', 'relpath', 'fname']
        addcols[:] = [c for c in addcols if c != 'dirname']
    else:
        firstcols = ['relpath', 'fname']
    mdcols = []  # Names of additional metadata columns from named captures.
    if dirpat is not None:
        dirpat = re.compile(dirpat)
        dirmdcols = list(dirpat.groupindex.keys())
        for c in dirmdcols:
            try:
                assert(c not in firstcols + addcols)
            except AssertionError:
                msg = 'Named group {:} masks another output column.'.format(c)
                raise RuntimeError(msg)
        mdcols += dirmdcols
    if fnpat is not None:
        fnpat = re.compile(fnpat)
        fnmdcols = list(fnpat.groupindex.keys())
        for c in fnmdcols:
            try:
                assert(c not in firstcols + addcols + mdcols)
            except AssertionError:
                msg = 'Named group {:} masks another output column.'.format(c)
                raise RuntimeError(msg)
        mdcols += fnmdcols

    stats = {'bytes': 'st_size'} if 'bytes' in addcols else {}
    if 'mtime' in addcols:
        stats['mtime'] = 'st_mtime'

    if dotdirs is False:
        try:
            assert(kwargs['topdown'] is True)
        except AssertionError:
            msg = 'topdown=False not compatible with dotdirs=False'
            raise RuntimeError(msg)
        except KeyError:
            pass

    recs = []
    for root, dirs, files in os.walk(dirname, **kwargs):
        if sentinel in files:
            dirs[:] = []  # Do not descend into subdirectories.
            continue      # Do not include files in this directory.
        relpath = os.path.relpath(root, dirname)
        dircols = []
        if dirpat is not None:
            dirm = dirpat.search(relpath)
            if dirm is None:
                continue # Do not include files in this directory.
            # Add named capture groups and replace unmatched optional
            # named captures with empty string.
            dircols = [
                '' if s is None else s for s in dirm.groupdict().values()
            ]
        for name in files:
            if (dotfiles is False) and (name[0] == '.'):
                continue
            fncols = []
            if fnpat is not None:
                fnm = fnpat.search(name)
                if fnm is None:
                    continue # Do not include this file.
                # Add named capture groups and replace unmatched optional
                # named captures with empty string.
                fncols = [
                    '' if s is None else s for s in fnm.groupdict().values()
                ]
            reccols = []
            if stats != {}:
                st = os.stat(os.path.join(root, name))
            if ('barename' in addcols) or ('ext' in addcols):
                (barename, ext) = os.path.splitext(name)
            for col in firstcols + addcols:
                if col in ['bytes', 'mtime']:
                    reccols += [getattr(st, stats[col])]
                elif col == 'relpath':
                    reccols += [relpath]
                elif col == 'fname':
                    reccols += [name]
                elif col == 'dirname':
                    reccols += [dirname]
                elif col == 'barename':
                    reccols += [barename]
                elif col == 'ext':
                    reccols += [ext]
            recs.append(reccols + dircols + fncols)

        # Change dirs in-place to prevent os.walk() from descending into
        # '.' directories.
        if dotdirs is False:
            dirs[:] = [d for d in dirs if not d[0] == '.']

    df = pd.DataFrame(
        recs,
        columns=firstcols + addcols + mdcols
    )
    df = df.sort_values(
        by=['relpath', 'fname'], axis='rows'
    ).reset_index(drop=True)
    if len(df) > 0:
        if 'dirname' in firstcols:
            df.dirname = df.dirname.astype('category')
        df.relpath = df.relpath.astype('category')
        df.fname = df.fname.astype('category')
        if 'barename' in addcols:
            df.barename = df.barename.astype('category')
        if 'ext' in addcols:
            df.ext = df.ext.astype('category')
        # Cast named captured columns to Categorical.
        for col in mdcols:
            df.loc[:, col] = df.loc[:, col].astype('category')
        if to_datetime is True and 'mtime' in df.columns:
            df.loc[:, 'mtime'] = pd.to_datetime(df.loc[:, 'mtime'], unit='s')
    return df

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
