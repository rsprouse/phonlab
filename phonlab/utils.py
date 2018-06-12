# Utility functions.

import os
import pandas as pd
import re

def dir2df(dirname, addcols=[], fnpat=None, dirpat=None, sentinel='.bad.txt',
to_datetime=True, dotfiles=False, dotdirs=False, **kwargs):
    '''dir2df(): Recursively generate the filenames in a directory tree
using os.walk() and store as rows of a DataFrame.

'Hidden' files and directories (those with names that start with '.') are
ignored by default. dir2df() will also not descend into a directory tree that
contains a sentinel file (default name '.bad.txt').


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

stats : str, sequence of str
    If not None, add a column for any of the `st_*` attributes in the
    `stat` structure returned by os.stat(). For example,
    `stats=['size', 'mtime']` returns file size in bytes and last modification
    times from `st_size` and `st_mtime`. If 'size' is requested, the added
    column is named 'bytes' to avoid conflicting with the DataFrame 'size'
    attribute.  

    The time-based stats are cast to Pandas Timestamps automatically unless
    `to_datetime` is False. Resolution of the time-based stats is dependent
    on your platform; see the os.stat() documentation.

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

to_datetime : boolean (default True)
    If True, any time-based stats (ones that end in `time` or `time_ns`)
    will be converted from Unix epoch to datetime. If False, the values
    will not be converted.

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
        # Cast named captured columns to Categorical.
        for col in mdcols:
            df.loc[:, col] = df.loc[:, col].astype('category')
        if (to_datetime is True) and (stats is not None):
            for s in stats:
                # Use string slice notation [3:] to trim 'st_' prefix.
                s = s[3:]
                if s.endswith('time_ns'):
                    df.loc[:, s] = pd.to_datetime(df.loc[:, s], unit='ns')
                elif s.endswith('time'):
                    df.loc[:, s] = pd.to_datetime(df.loc[:, s], unit='s')
    return df
