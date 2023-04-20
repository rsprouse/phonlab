import numpy as np
import pandas as pd

def formant2df(obj, num, ts=None, unit='HERTZ', include_bw=False, tcol='sec'):
    '''
    Return formant values from a Praat Formant object as a dataframe.

    Parameters
    ----------

    obj: Formant obj
    The input Formant object.

    num: int or list of int
    The formants from which values will be returned. If `num` is `int` type
    the values of the first `num` formants are returned. If `num` is a list
    of `int`s then the exactly the formants in the list are returned.

    ts: Iterable of floats or None (default)
    If None (default), then the Formant object's `ts` times are used to
    query for formant measurements. These are the centers of the analysis
    frames. The Formant object's values can also be queried at specific
    times by providing an Iterable of floats, such as a numpy array or
    Python list.

    unit: str 'HERTZ' (default) or 'BARK'
    The unit in which formant values are returned. Lower case versions of
    the units are allowed. See Praat's `Formant_enums.h` file for defined
    values.

    include_bw: boolean (default False)
    If True, bandwidth values for each formant returned are also included.

    tcol: str (default 'sec') or None
    The label of the time column in the output dataframe, 'sec' by default.
    If None, no time column is added to the dataframe.

    Returns
    -------

    DataFrame
    The output dataframe contains columns of times and formant measurements.
    The formant columns are labelled `fN`, where `N` is an integer. The time
    column is labelled by the value of `tcol` (default 'sec'), or omitted if
    `tcol` is None. If `include_bw` is True then bandwidths columns labelled
    `bwN` corresponding to `fN` columns are also returned.

    Examples
    --------

    >>> snd = parselmouth.Sound(mywav)
    >>> fmnt = snd.to_formant_burg()

    # Get dataframe of first two formants in Hz and corresponding bandwidths.
    >>> f12df = formant2df(fmnt, 2, include_bw=True)

    # Get dataframe of only second formant in bark and no bandwidths.
    >>> f1df = formant2df(fmnt, [2], unit='bark')

    # Save results to file the same way you would any other dataframe.
    >>> f1df.to_csv('formants.csv', sep='\t', header=True, index=False)
    >>> f1df.to_pickle('formants.zip')
    '''
    fNs = np.arange(1, num+1) if isinstance(num, int) else num
    ts = obj.ts() if ts is None else ts
    data = {
        f'f{fn}': \
            np.array(
                [obj.get_value_at_time(fn, t, unit.upper()) for t in ts]
            ) for fn in fNs
    }
    if include_bw is True:
        data.update({
            f'bw{fn}': \
                np.array(
                    [obj.get_bandwidth_at_time(fn, t) for t in ts]
                ) for fn in fNs
        })
    if tcol is None:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame({**{tcol: ts}, **data})

def pitch2df(obj, ts=None, unit='HERTZ', interpolation='LINEAR', tcol='sec'):
    '''
    Return pitch values from a Praat Pitch object as a dataframe.

    Parameters
    ----------

    obj: Pitch obj
    The input Pitch object.

    ts: Iterable of floats or None (default)
    If None (default), then the Pitch object's `ts` times are used to
    query for pitch measurements. These are the centers of the analysis
    frames. The Pitch object's values can also be queried at specific
    times by providing an Iterable of floats, such as a numpy array or
    Python list.

    unit: str 'HERTZ' (default), 'HERTZ_LOGARITHMIC', 'MEL', 'LOG_HERTZ',
    'SEMITONES_1', 'SEMITONES_100', 'SEMITONES_200', 'SEMITONES_440', 'ERB'
    The unit in which pitch values are returned. Lower case versions of
    the units are allowed. See Praat's `Pitch_enums.h` file for defined
    values.

    interpolation: str 'LINEAR' (default), 'NEAREST'
    The type of interpolation to use when returning values. Lower case
    versions of the units are allowed. See Praat's `Pitch.cpp` and
    `Vector_enums.h` files for defined values.

    tcol: str (default 'sec') or None
    The label of the time column in the output dataframe, 'sec' by default.
    If None, no time column is added to the dataframe.

    Returns
    -------

    DataFrame
    The output dataframe contains a column of pitch measurements labelled
    `f0`. The time column is labelled by the value of `tcol` (default 'sec'),
    or omitted if `tcol` is None.

    Examples
    --------

    >>> snd = parselmouth.Sound(mywav)
    >>> ptch = snd.to_pitch()

    # Get dataframe of pitch in Hz.
    >>> hzdf = pitch2df(pitch)
    
    # Get dataframe of pitch in mel.
    >>> meldf = pitch2df(pitch, unit='mel')

    # Save results to file the same way you would any other dataframe.
    >>> hzdf.to_csv('hzpitch.csv', sep='\t', header=True, index=False)
    >>> hzdf.to_pickle('melpitch.zip')
    '''
    ts = obj.ts() if ts is None else ts
    data = {
        'f0': \
            np.array(
                [
                    obj.get_value_at_time(
                        t, unit.upper(), interpolation.upper()
                    ) for t in ts
                ]
            )
    }
    if tcol is None:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame({**{tcol: ts}, **data})

def intensity2df(obj, ts=None, interpolation='CUBIC', tcol='sec'):
    '''
    Return intensity values from a Praat Intensity object as a dataframe.

    Parameters
    ----------

    obj: Intensity obj
    The input Intensity object.

    ts: Iterable of floats or None (default)
    If None (default), then the Intensity object's `ts` times are used to
    query for formant measurements. These are the centers of the analysis
    frames. The Intensity object's values can also be queried at specific
    times by providing an Iterable of floats, such as a numpy array or
    Python list.

    interpolation: str 'CUBIC' (default), 'NEAREST', 'LINEAR', 'SINC70',
    'SINC700'
    The type of interpolation to use when returning values. Lower case
    versions of the units are allowed. See Praat's `Intensity.cpp` and
    `Vector_enums.h` files for defined values.

    tcol: str (default 'sec') or None
    The label of the time column in the output dataframe, 'sec' by default.
    If None, no time column is added to the dataframe.

    Returns
    -------

    DataFrame
    The output dataframe contains a column of intensity measurements labelled
    `spl`. The time column is labelled by the value of `tcol` (default 'sec'),
    or omitted if `tcol` is None.

    Examples
    --------

    >>> snd = parselmouth.Sound(mywav)
    >>> intens = snd.to_intensity()

    # Get dataframe of SPL.
    >>> spldf = intensity2df(intens)

    # Save results to file the same way you would any other dataframe.
    >>> spldf.to_csv('intensity.csv', sep='\t', header=True, index=False)
    >>> splddf.to_pickle('intensity.zip')
    '''
    ts = obj.ts() if ts is None else ts
    data = {
        'spl': \
            np.array(
                [obj.get_value(t, interpolation.upper()) for t in ts]
            )
    }
    if tcol is None:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame({**{tcol: ts}, **data})
