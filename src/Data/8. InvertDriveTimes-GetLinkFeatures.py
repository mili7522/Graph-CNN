import pandas as pd
import numpy as np

driveTimes = pd.read_csv('2018-06-01-NSW-NeighbourDriveTimesAndDistance.csv')
distance = driveTimes.copy()
neighbours = pd.read_csv('Geography/2018-06-01-NSW-Neighbouring_Suburbs_With_Bridges-GCC.csv')

#driveTimes['Time (min)'][driveTimes['Time (min)'] < 10].hist()
#driveTimes['Time (min)'].mean()
#driveTimes['Time (min)'].median()

clipedDriveTimes = np.clip(driveTimes['Time (min)'], 1, 120)  # Clip between 1 minute and 120 minutes

inverseDriveTimes = 1 / clipedDriveTimes

maxInverseDriveTime = inverseDriveTimes.max()
minInverseDriveTime = inverseDriveTimes.min()

normInverseDriveTimes = inverseDriveTimes / maxInverseDriveTime

neighbours['Inverse Times'] = normInverseDriveTimes


# Distances

clipedDistances = np.clip(distance['Distance'], 0.1, None)  # Clip below 0.1

inverseDistance = 1 / clipedDistances

maxInverseDistance = inverseDistance.max()
minInverseDistance = inverseDistance.min()

normInverseDistance = inverseDistance / maxInverseDistance

neighbours['Inverse Distance'] = normInverseDistance


# Save
neighbours.to_csv('2018-06-01-NSW-NeighbourLinkFeatures.csv', index = False)



### VIC

driveTimes = pd.read_csv('2018-06-07-VIC-NeighbourDriveTimesAndDistance.csv')
distance = driveTimes.copy()
neighbours = pd.read_csv('Geography/2018-06-07-VIC-Neighbouring_Suburbs_With_Bridges-GCC.csv')


clipedDriveTimes = np.clip(driveTimes['Time (min)'], 1, 120)  # Clip between 1 minute and 120 minutes

inverseDriveTimes = 1 / clipedDriveTimes

maxInverseDriveTime = inverseDriveTimes.max()
minInverseDriveTime = inverseDriveTimes.min()

normInverseDriveTimes = inverseDriveTimes / maxInverseDriveTime

neighbours['Inverse Times'] = normInverseDriveTimes


# Distances

clipedDistances = np.clip(distance['Distance'], 0.1, None)  # Clip below 0.1

inverseDistance = 1 / clipedDistances

maxInverseDistance = inverseDistance.max()
minInverseDistance = inverseDistance.min()

normInverseDistance = inverseDistance / maxInverseDistance

neighbours['Inverse Distance'] = normInverseDistance


# Save
neighbours.to_csv('2018-06-07-VIC-NeighbourLinkFeatures.csv', index = False)
