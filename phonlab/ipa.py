# Utility functions.

import pandas as pd
import pkg_resources

'''Dataframe containing mapping of ARAPABET to Unicode IPA.'''
arpabet2ipa = pd.read_csv(
    pkg_resources.resource_stream('phonlab', 'data/arpabet2ipa.csv')
)

