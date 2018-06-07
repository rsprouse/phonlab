# Utility functions.

import os
import pandas as pd
import re

def dir2df(dirname, fnpat=None, dirpat=None, stats=None, sentinel='.bad.txt',
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
    if fnpat is not None:
        fnpat = re.compile(fnpat)
    if dirpat is not None:
        dirpat = re.compile(dirpat)

    # Cast stats to a list, if str.
    stats = [stats] if stats == str(stats) else stats
    if stats is not None:
        # Handle 'bytes' as an alias for 'size'
        stats = [s if s != 'bytes' else 'size' for s in stats]
        # Add 'st_' prefix, if necessary.
        stats = [s if s.startswith('st_') else 'st_' + s for s in stats]

    if dotdirs is False:
        try:
            assert(kwargs['topdown'] is True)
        except AssertionError:
            msg = 'topdown=False not compatible with dotdirs=False'
            raise RuntimeError(msg)
        except KeyError:
            pass

    recs = []
    namedcaptures = []
    for root, dirs, files in os.walk(dirname, **kwargs):
        if sentinel in files:
            dirs[:] = []  # Do not descend into subdirectories.
            continue      # Do not include files in this directory.
        relpath = os.path.relpath(root, dirname)
        dircols = {}
        if dirpat is not None:
            dirm = dirpat.search(relpath)
            if dirm is None:
                continue # Do not include files in this directory.
            elif len(dirm.groupdict()) > 0:
                # Add named capture groups and replace unmatched optional
                # named captures with empty string.
                for k, v in dirm.groupdict().items():
                    dircols[k] = v if v is not None else ''
        for name in files:
            fncols = {}
            if fnpat is not None:
                m = fnpat.search(name)
                if m is None:
                    continue # Do not include this file.
                elif len(m.groupdict()) > 0:
                    # Add named capture groups and replace unmatched optional
                    # named captures with empty string.
                    for k, v in m.groupdict().items():
                        fncols[k] = v if v is not None else ''
            if (dotfiles is False) and (name[0] == '.'):
                continue
            if namedcaptures == []:
                namedcaptures = list(dircols.keys()) + list(fncols.keys())
            rec = [
                relpath,
                name
            ] + list(dircols.values()) + list(fncols.values())
            if stats is not None:
                st = os.stat(os.path.join(root, name))
                for attr in stats:
                    val = getattr(st, attr)
                    # Rename 'size' to avoid conflict with df size attribute.
                    if attr == 'st_size': 
                        attr = 'st_bytes'
                    # Use string slice notation [3:] to trim 'st_' prefix.
                    rec[attr[3:]] = val
            recs.append(rec)

        # Change dirs in-place to prevent os.walk() from descending into
        # '.' directories.
        if dotdirs is False:
            dirs[:] = [d for d in dirs if not d[0] == '.']

    df = pd.DataFrame(recs, columns=['relpath', 'filename'] + namedcaptures)
    df = df.sort_values(
        by=['relpath', 'filename'], axis='rows'
    ).reset_index(drop=True)
    if len(df) > 0:
        df.relpath = df.relpath.astype('category')
        # Cast named captured columns to Categorical.
        for col in namedcaptures:
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
