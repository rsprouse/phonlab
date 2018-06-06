# Utility functions.

import os
import pandas as pd
import re

def dir2df(dirname, fnpat=None, dirpat=None, stats=None,
to_datetime=True, dotfiles=False, dotdirs=False, **kwargs):
    '''dir2df(): Recursively generate the filenames in a directory tree
using os.walk() and store as rows of a DataFrame. With default parameter
values 'hidden' files and directories (those with names that start with `.`)
are ignored.


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

    If you use named captures in `fnpat`, new columns will appear in
    the output that correspond to the capture group contents.

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
    times from `st_size` and `st_mtime`. The time-based stats are cast to
    Pandas Timestamps automatically unless `to_datetime` is False. Resolution
    of the time-based stats is dependent on your platform; see the os.stat()
    documentation.

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
        relpath = os.path.relpath(root, dirname)
        dircols = {}
        if dirpat is not None:
            dirm = dirpat.search(relpath)
            if dirm is None:
                continue
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
                    continue
                elif len(m.groupdict()) > 0:
                    # Add named capture groups and replace unmatched optional
                    # named captures with empty string.
                    for k, v in m.groupdict().items():
                        fncols[k] = v if v is not None else ''
            if (dotfiles is False) and (name[0] == '.'):
                continue
            rec = {
                **dircols,
                **fncols,
                'filename': name,
                'relpath': relpath
            }
            if stats is not None:
                st = os.stat(os.path.join(root, name))
                for attr in stats:
                    stattr = attr
                    if not stattr.startswith('st_'):
                        stattr = 'st_' + stattr
                    rec[attr] = getattr(st, stattr)
            recs.append(rec)

        # Change dirs in-place to prevent os.walk() from descending into
        # '.' directories.
        if dotdirs is False:
            dirs[:] = [d for d in dirs if not d[0] == '.']

    df = pd.DataFrame.from_records(recs)
    if len(df) > 0:
        df.relpath = df.relpath.astype('category')
        if (to_datetime is True) and (stats is not None):
            for s in stats:
                if s.endswith('time_ns'):
                    df.loc[:, s] = pd.to_datetime(df.loc[:, s], unit='ns')
                elif s.endswith('time'):
                    df.loc[:, s] = pd.to_datetime(df.loc[:, s], unit='s')
    return df
