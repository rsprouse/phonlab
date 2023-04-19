import numpy as np
import pandas as pd

def formant2df(fobj, num, unit='HERTZ', include_bw=False):
    '''
    Return formant values from a Praat Formant object as a dataframe.

    Parameters
    ----------

    fobj: Formant obj
    The input Formant object.

    num: int or list of int
    The formants from which values will be returned. If `num` is `int` type
    the values of the first `num` formants are returned. If `num` is a list
    of `int`s then the exactly the formants in the list are returned.

    unit: str 'HERTZ' (default) or 'BARK'
    The unit in which formant values are returned. Lower case versions of
    the units are allowed. See Praat's `Formant_enums.h` file for defined
    values.

    include_bw: boolean, default False
    If True, bandwidth values for each formant returned are also included.

    Returns
    -------

    DataFrame
    The output dataframe contains columns of times and formant measurements.
    The time column is labelled `sec` and formant columns are labelled `fN`,
    where `N` is an integer. If `include_bw` is True then bandwidths columns
    labelled `bwN` corresponding to `fN` columns are also returned.

    Examples
    --------

    snd = parselmouth.Sound(mywav)
    fmnt = snd.to_formant_burg()

    # Get dataframe of first two formants in Hz and corresponding bandwidths.
    f12df = formant2df(fmnt, 2, include_bw=True)

    # Get dataframe of only second formant in bark and no bandwidths.
    f1df = formant2df(fmnt, [2], unit='bark')

    # Save results to file the same way you would any other dataframe.
    f1df.to_csv('formants.csv', sep='\t', header=True, index=False)
    f1df.to_pickle('formants.zip')
    '''
    fNs = np.arange(1, num+1) if isinstance(num, int) else num
    data = {
        f'f{fn}': \
            np.array(
                [fobj.get_value_at_time(fn, t, unit.upper()) for t in fobj.ts()]
            ) for fn in fNs
    }
    if include_bw is True:
        data.update({
            f'bw{fn}': \
                np.array(
                    [fobj.get_bandwidth_at_time(fn, t) for t in fobj.ts()]
                ) for fn in fNs
        })
    return pd.DataFrame({**{'sec': fobj.ts()}, **data})

def pitch2df(pobj, unit='HERTZ'):
    '''
    Return formant values from a Praat Pitch object as a dataframe.

    Parameters
    ----------

    pobj: Pitch obj
    The input Pitch object.

    unit: str 'HERTZ' (default), 'HERTZ_LOGARITHMIC', 'MEL', 'LOG_HERTZ',
    'SEMITONES_1', 'SEMITONES_100', 'SEMITONES_200', 'SEMITONES_440', 'ERB'
    The unit in which pitch values are returned. Lower case versions of
    the units are allowed. See Praat's `Pitch_enums.h` file for defined
    values.

    Returns
    -------

    DataFrame
    The output dataframe contains columns of times and pitch measurements.
    The time column is labelled `sec` and the pitch column is labelled `f0`.

    Examples
    --------

    snd = parselmouth.Sound(mywav)
    ptch = snd.to_pitch()

    # Get dataframe of pitch in Hz.
    hzdf = pitch2df(pitch)
    
    # Get dataframe of pitch in mel.
    meldf = pitch2df(pitch, unit='mel')

    # Save results to file the same way you would any other dataframe.
    hzdf.to_csv('hzpitch.csv', sep='\t', header=True, index=False)
    hzdf.to_pickle('melpitch.zip')
    '''
    frameidx = np.arange(1, pobj.nx+1)
    data = {
        'sec': pobj.ts(),
        'f0': \
            np.array(
                [pobj.get_value_in_frame(n, unit.upper()) for n in frameidx]
            )
    }
    return pd.DataFrame(data)

