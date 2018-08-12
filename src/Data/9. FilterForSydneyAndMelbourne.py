import pandas as pd

SA1s = pd.read_csv('Geography/SA1_2016_AUST.csv')

sydneySA1s = SA1s[SA1s['GCCSA_NAME_2016'] == 'Greater Sydney']['SA1_7DIGITCODE_2016']


# inputs = pd.read_csv('2018-06-01-NSW-SA1Input-Normalised.csv')
# sydney_inputs = inputs[inputs['SA1_7DIGITCODE_2016'].isin(sydneySA1s)]
# sydney_inputs.to_csv('2018-06-03-SYD-SA1Input-Normalised.csv', index = False)

# linkFeatures = pd.read_csv('2018-06-01-NSW-NeighbourLinkFeatures.csv')
# sydney_linkFeatures = linkFeatures[linkFeatures['src_SA1_7DIG16'].isin(sydneySA1s) & linkFeatures['nbr_SA1_7DIG16'].isin(sydneySA1s)]
# sydney_linkFeatures.to_csv('2018-06-03-SYD-NeighbourLinkFeatures.csv', index = False)


inputs = pd.read_csv('2018-08-02-NSW-SA1Input-Normalised-RealValues.csv')
sydney_inputs = inputs[inputs['SA1_7DIGITCODE_2016'].isin(sydneySA1s)]
sydney_inputs.to_csv('2018-06-03-SYD-SA1Input-Normalised-RealValues.csv', index = False)

### VIC

# melbourneSA1s = SA1s[SA1s['GCCSA_NAME_2016'] == 'Greater Melbourne']['SA1_7DIGITCODE_2016']


# inputs = pd.read_csv('2018-06-07-VIC-SA1Input-Normalised.csv')
# melbourne_inputs = inputs[inputs['SA1_7DIGITCODE_2016'].isin(melbourneSA1s)]
# melbourne_inputs.to_csv('2018-06-07-MEL-SA1Input-Normalised.csv', index = False)

# linkFeatures = pd.read_csv('2018-06-07-VIC-NeighbourLinkFeatures.csv')
# melbourne_linkFeatures = linkFeatures[linkFeatures['src_SA1_7DIG16'].isin(melbourneSA1s) & linkFeatures['nbr_SA1_7DIG16'].isin(melbourneSA1s)]
# melbourne_linkFeatures.to_csv('2018-06-07-MEL-NeighbourLinkFeatures.csv', index = False)