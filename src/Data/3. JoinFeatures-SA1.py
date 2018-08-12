import pandas as pd
import numpy as np

SA1s = pd.read_csv('Geography/SA1_2016_AUST.csv')
SA1s.set_index('SA1_7DIGITCODE_2016', inplace = True)

SA1_FEATURES = pd.read_csv('SA1_FEATURES_AU_RATIO.csv', na_values = ['#VALUE!', '#DIV/0!'])
                                                             
SA1_FEATURES.set_index('SA1_7DIGITCODE_2016', inplace = True)
SA1_FEATURES.dropna(0, 'any', inplace = True)


vicSA1s = SA1_FEATURES.loc[SA1s['GCCSA_NAME_2016'].isin(['Rest of Vic.', 'Greater Melbourne'])]
nswSA1s = SA1_FEATURES.loc[SA1s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])]


# Load AEC Data (Vic)
AEC = pd.read_csv('AEC_PP_TCP_FINAL_AU.csv')
AEC.set_index('SA1_id', inplace = True)

vic_AEC = AEC.loc[vicSA1s.index]
vic_AEC.dropna(0, 'any', inplace = True)


# Load AEC Data (NSW)
AEC = pd.read_csv('AEC_PP_TCP_NSW.csv')
AEC.set_index('SA1_id', inplace = True)
nsw_AEC = AEC.loc[nswSA1s.index]

nsw_AEC.dropna(0, 'any', inplace = True)


# Filter the features again, dropping any missing AEC values
vicSA1s = vicSA1s.loc[vic_AEC.index]
nswSA1s = nswSA1s.loc[nsw_AEC.index]


vicSA1s['Category'] = (vic_AEC['vote_pc'] // 10).astype(int)
nswSA1s['Category'] = np.floor(nsw_AEC['vote_pc'] * 10).astype(int)
vicSA1s.to_csv('2018-06-07-VIC-SA1Input.csv', index = True)
nswSA1s.to_csv('2018-06-01-NSW-SA1Input.csv', index = True)
