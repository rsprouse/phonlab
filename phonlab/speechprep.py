import sys
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile as tmpf
import phonlab
import soundfile

import numpy.random as random
try:
    from importlib.resources import files as res_files
    from importlib.resources import as_file
except ImportError:
    from importlib_resources import files as res_files
    from importlib_resources import as_file

def _prep_noise_file(nsrcfile, normfile, compinfo, noisedb, soxcmd='sox'):
    '''
    Prepare normalized noise file that matches a comparison signal file.
    
    Parameters
    ----------
    
    nsrcfile : path-like
        The input noise file to normalize.
    normfile : path-like
        The output normalized noise file.
    compinfo : _SoundFileInfo
        _SoundFileInfo object that contains file parameters that the
        output normalized noise file must match: duration, samplerate,
        (number of) channels.
    noisedb : number
        The full scale gain-dB value used to scale the noise.
    soxcmd : path-like (default 'sox')
        The system command used to invoke sox. By default whichever `sox` exists
        first in the path will be used.

    Returns
    -------
    
    No value is returned from this function. Failure to create an output file raises
    an error.
    '''
    tgtdur = compinfo.duration # Output duration
    nsrcinfo = soundfile.info(nsrcfile)
    try:
        assert(compinfo.channels in (1, 2))
    except AssertionError:
        msg = f'Signal file must have 1 or 2 channels. ' \
              f'Found {compinfo.channels}.'
        raise RuntimeError(msg)
    try:
        assert(compinfo.channels == nsrcinfo.channels)
    except AssertionError:
        msg = f'Noise channel count ({nsrcinfo.channels}) does not match ' \
              f'channel count for signal file ({compinfo.channels}).'
        raise RuntimeError(msg)
    if nsrcinfo.samplerate < compinfo.samplerate:
        msg = f'Upsampling noise file {nsrcfile} from {nsrcinfo.samplerate} ' \
              f'to {compinfo.samplerate}.'
        sys.stderr.write(msg)
    if nsrcinfo.duration < tgtdur:
        msg = 'Noise file too short. Automatic lengthening of noise file not implemented. '
        raise NotImplementedError(msg)
    else:
        start = (nsrcinfo.duration - tgtdur) * random.random()
    with tmpf(suffix=Path(normfile).suffix) as noiseout:
        proc = subprocess.run(
            [
                str(soxcmd),
                str(nsrcfile),
                '-r', str(compinfo.samplerate), str(normfile),
                'trim', str(start), f'+{tgtdur}',
                'norm', str(noisedb)
            ]
        )

def add_noise_and_norm(infile, outfile, noise, noisedb, normdb, soxcmd='sox'):
    '''
    Add noise to an audio file and normalize the result before writing to a file.
    The input signal audio is normed to (nearly) 0dB, and the noise is normed to
    `noisedb` before mixing. The mixed output signal is normed to `normdb`.

    Parameters
    ----------
    
    infile : path-like
        The input audio signal filename. Must be a mono or stereo file of a type
        that sox can handle.
    outfile : path-like
        The output audio filename. Must be a type that sox can handle.
    noise : str or path-like
        The kind of noise to add to the signal file. Three kinds are allowed:
            1. Noises that can be generated by the sox `synth` effect:
                   'sine', 'square', 'triangle', 'sawtooth', 'trapezium', 'exp',
                   'noise', 'whitenoise', 'tpdfnoise', 'pinknoise', 'brownnoise',
                   'pluck'
            2. Pre-generated noises provided by .wav files in this package:
               'babble', 'party', 'restaurant'
            3. Filename path of any pre-generated noise file created by the user.
    noisedb : number
        The full scale gain-dB value used to scale the noise to be mixed with the signal.
        0dB is full scale, -3dB is approximately 70% of linear, -6dB is approximately
        50% of linear. This value is passed to the sox `norm -n` effect.
    normdb : number
        The full scale gain-dB value used to norm the output file after mixing the
        signal and noise.
    soxcmd : path-like (default 'sox')
        The system command used to invoke sox. By default whichever `sox` exists
        first in the path will be used.

    Returns
    -------
    
    No value is returned from this function. Failure to create an output file raises
    an error.

    Examples
    --------
    
    # Create an output file named `signal.party.-3.0.-1.0.wav` that contains
    # 'party' noise at -3dB and with the output file normed to -1dB.
    >>> noisedb = -3.0
    >>> normdb = -1.0
    >>> noise = 'party'
    >>> infile = Path('signal.wav')
    >>> outfile = infile.with_suffix(f'.{noise}.{noisedb}.{normdb}.wav')
    >>> add_noise_and_norm(infile, outfile, noisedb=noisedb, noise=noise, normdb=normdb)
    
    # Create an output file named `signal.brownnoise.-3.0.-1.0.wav` that contains
    # 'brownnoise' noise at -3dB and with the output file normed to -1dB.
    >>> noise = 'brownnoise'
    >>> outfile = infile.with_suffix(f'.{noise}.{noisedb}.{normdb}.wav')
    >>> add_noise_and_norm(infile, outfile, noisedb=noisedb, noise=noise, normdb=normdb)
    
    # Create an output file named `signal.mynoise.-3.0.-1.0.wav` that contains
    # pre-generated noise at -3dB from the file `mynoise.wav` in the current
    # working directory and with the output file normed to -1dB.
    >>> noise = 'mynoise.wav'
    >>> outfile = infile.with_suffix(f'.{Path(noise).stem}.{normdb}.wav')
    >>> add_noise_and_norm(infile, outfile, noisedb=noisedb, noise=noise, normdb=normdb)    
    
    '''
    # Valid options that can be passed to the `sox` `synth` effect.
    sox_noise = (
        'sine', 'square', 'triangle', 'sawtooth', 'trapezium', 'exp',
        'noise', 'whitenoise', 'tpdfnoise', 'pinknoise', 'brownnoise', 'pluck'
    )
    # Names of files in the package data/noise directory. These must be in pairs
    # of the form {noise}-mono.wav and {noise}-stereo.wav.
    pkg_noise = (
        'babble', 'party', 'restaurant'
    )

    sfx = Path(infile).suffix
    with tmpf(suffix=sfx) as normsig, tmpf(suffix=sfx) as normnoise:

        # Normalize signal to (nearly) full scale. Actual full scale
        # using '0' as the norm value can result in clipping warnings
        # for a small number of samples.
        subprocess.run(
            [str(soxcmd), infile, normsig.name, 'norm', '-0.001']
        )

        # Open noise file or synthesize noise; normalize to desired level.
        if noise in sox_noise:
            subprocess.run(
                [
                    str(soxcmd),
                    str(infile),
                    normnoise.name,
                    'synth', str(noise),
                    'norm', str(noisedb)
                ]
            )
        else:
            compinfo = soundfile.info(infile)  # Comparison info
            if noise in pkg_noise:
                chan = {1: 'mono', 2: 'stereo'}[compinfo.channels]
                noisesrc = f'{noise}-{chan}.wav'
                respath = res_files('phonlab') / 'data' / 'noise' / noisesrc
                with as_file(respath) as noisepath:
                    _prep_noise_file(
                        noisepath,
                        normnoise.name,
                        compinfo,
                        noisedb,
                        soxcmd
                    )
            else:
                _prep_noise_file(noise, normnoise.name, compinfo, noisedb, soxcmd)

        # Mix signal and noise; normalize to desired level.
        subprocess.run(
            [
                str(soxcmd),
                '-m', normsig.name, normnoise.name,
                str(outfile),
                'norm', str(normdb)
            ]
        )
