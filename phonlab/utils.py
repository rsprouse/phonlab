# Utility functions.

import os
import pandas as pd
import re

def dir2df(dirname, searchpat=None, dirpat=None, dotfiles=False,
dotdirs=False, stats=None, to_datetime=True, **kwargs):
    '''Recursively generate the filenames in a directory tree using os.walk()
and store in a DataFrame. With default parameter values 'hidden' files and
directories (those with names that start with `.`) are ignored.


Parameters
----------

dirname : str
    Name of directory to search for filenames.


Optional parameters
-------------------

searchpat : str, re
    Regular expression pattern that defines the filenames to return.
    The only filenames in the result set will be those that return a match
    for `re.search(searchpat, filename)`.

    If you use named captures in `searchpat`, new columns will appear in
    the output that correspond to the capture group contents.

    If you need to use a flag with your pattern, you can use a precompiled
    regex for the value of `searchpat`. For example, you can do
    case-insensitive matching of '.wav' and '.WAV' files with
    `re.compile(r'.wav$', re.IGNORECASE)`.

dirpat : str, re
    Like searchpat, only applied against the relative path in dirname.
    Relative paths that do not match `dirpat` will be skipped.

dotfiles : boolean (default False)
    If True, include filenames beginning with `.` in the output. Otherwise,
    omit these names.

dotdirs : boolean (default False)
    If True, descend into directories with names that begin with `.`. If
    False, skip these directories.

stats : str, sequence of str
    If not None, add a column for any of the `st_*` attributes returned in the
    `stat` structure returned by os.stat(). For example,
    `stats=['size', 'mtime']` returns file size in bytes and last modification
    times from `st_size` and `st_mtime`. The time-based stats are cast to
    Pandas Timestamps automatically. Resolution of the time-based stats is
    dependent on your platform; see the os.stat() documentation.

to_datetime : boolean (default True)
    If True, any time-based stats (ones that end in `time` or `time_ns`)
    will be converted from Unix epoch to datetime. If False, the values
    will not be converted.

kwargs : various
    Remaining kwargs are passed to os.walk(). If not used, then os.walk() will
    be called with default kwargs. Note that using os.walk(topdown=False) is
    not compatible with `dotdirs=False`.


Returns
-------

dirdf : DataFrame
    Pandas DataFrame with filenames recorded in rows.

'''
    if searchpat is not None:
        searchpat = re.compile(searchpat)
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
        dircols = {}
        if dirpat is not None:
            dirm = dirpat.search(root)
            if dirm is None:
                continue
            elif len(dirm.groupdict()) > 0:
                # Add named capture groups and replace unmatched optional
                # named captures with empty string.
                for k, v in dirm.groupdict().items():
                    dircols[k] = v if v is not None else ''
        for name in files:
            patcols = {}
            if searchpat is not None:
                m = searchpat.search(name)
                if m is None:
                    continue
                elif len(m.groupdict()) > 0:
                    # Add named capture groups and replace unmatched optional
                    # named captures with empty string.
                    for k, v in m.groupdict().items():
                        patcols[k] = v if v is not None else ''
            if (dotfiles is False) and (name[0] == '.'):
                continue
            rec = {
                **dircols,
                **patcols,
                'filename': name,
                'relpath': os.path.relpath(root, dirname)
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
