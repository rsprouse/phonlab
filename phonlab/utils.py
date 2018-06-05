# Utility functions.

import os
import pandas as pd

def dir2df(dirname, endfilt=None, dotfiles=False, dotdirs=False, stats=None,
to_datetime=True, **kwargs):
    '''Recursively generate the filenames in a directory tree using os.walk()
and store in a DataFrame. With default parameter values 'hidden' files and
directories (those with names that start with `.`) are ignored.


Parameters
----------

dirname : str
    Name of directory to search for filenames.


Optional parameters
-------------------

endfilt : str, sequence of str (default None)
    Whitelist of filename endings to return in the result. If not None,
    ignore any filename that does not end with one of the strings defined in
    `endfilt`. For example, use `['.wav', '.WAV']` to restrict the output to
    only `.wav` and `.WAV` files.

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
    if endfilt is not None:
        # If endfilt is a str, cast to list of str so that
        # tuple(endfilt) does not break up the str into a sequence
        # of characters, e.g. '.wav' -> ('.', 'w', 'a', 'v').
        if endfilt == str(endfilt):
            endfilt = [endfilt]
        # Cast to tuple, since that's what ''.endswith() needs.
        endfilt = tuple(endfilt)

    # Cast stats to a list, if str.
    if stats is not None and stats == str(stats):
        stats = [stats]

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
        for name in files:
            if (endfilt is not None) and (not name.endswith(endfilt)):
                continue
            if (dotfiles is False) and (name[0] == '.'):
                continue
            rec = {
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

        # Remove '.' directories.
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
