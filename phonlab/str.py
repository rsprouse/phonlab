import os, sys, re
import pandas as pd
import pkg_resources

def phonemap(k, v, type='df'):
    '''Return a mapping between phones in different transcription systems as
a dataframe with each phoneset in a column. The rows contain correspondences
between phonesets.

Phone sets available for mapping are Unicode IPA ('ipa'), ARPABET ('arpabet'),
and GlobalPhone ('globalphone').

Parameters
----------

k : str (one of ['arpabet', 'globalphone', 'ipa']
    The name of the phoneset in the first column of the dataframe.

v : str (one of ['arpabet', 'globalphone', 'ipa']
    The name of the phoneset in the second column of the dataframe.

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


