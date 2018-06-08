import requests
import time
import pandas as pd
import numpy as np
from json import JSONDecodeError

def getLocstring(locations):
    if locations.ndim > 1:
        return ";".join([",".join(map(str, longlat)) for longlat in locations])
    else:
        return ",".join(map(str, locations))


def getDurations(origins, destinations = None):
    
    if destinations is None:
        locstring = getLocstring(origins)
        url = 'http://router.project-osrm.org/table/v1/driving/' + locstring
    else:
        locstring = getLocstring(origins) + ';' + getLocstring(destinations)
        url = 'http://router.project-osrm.org/table/v1/driving/' + locstring +\
                '?sources=' + ';'.join(map(str,range(len(np.atleast_2d(origins))))) +\
                '&destinations=' + ';'.join(map(str,range(len(np.atleast_2d(origins)), len(np.atleast_2d(origins)) + len(np.atleast_2d(destinations)))))
    
    response = requests.get(url)
    try:
        response = response.json()
    
        code = response['code']
        if code == 'Ok':
            durations = response['durations']
        else:
            print('Error')
            print(code)
            durations = None
    except:
        print('Error with requests.get')
        print(response)
        durations = None
        
    return durations

    
class durationGetter:
    def __init__(self, origins, destinations):
        assert len(origins) == len(destinations)
        self.origins = origins
        self.destinations = destinations
        self.completed = set()

        self.durations = np.zeros((len(origins), 1))
    
    
    def getDurations(self, max_i = None, sleepTimeForFails = 30):
        
        failed = []
        for i in range(len(self.origins)):
            if i in self.completed:
                continue
            
            print(i)
            trials = 0
            successfulGet = False
            while (not successfulGet) and (trials < 10):
                duration = getDurations(self.origins[i], self.destinations[i])
                if duration is not None:
                    successfulGet = True
                    self.durations[i] = duration[0]  # Get first (only) element
                    self.completed.add(i)
                else:
                    time.sleep(sleepTimeForFails)
            if not successfulGet:
                failed.append(i)
            
            if max_i is not None and i >= max_i:
                break
        
        if len(failed) > 0:
            print('Failed:')
            print(failed)

    
    def saveDurations(self, saveName, origins, destinations):
        if not saveName.endswith('.csv'):
            saveName += '.csv'
        df = pd.DataFrame(self.durations / 60, columns = ['Time (min)'])
        df['Origin'] = origins
        df['Destination'] = destinations
        df[['Origin', 'Destination', 'Time (min)']].to_csv(saveName, index = None)




links = pd.read_csv('Geography/2018-06-07-VIC-Neighbouring_Suburbs_With_Bridges-GCC.csv')
centres = pd.read_csv('Geography/2018-06-07-VIC-SA1-2016Centres.csv', index_col = 0)


origins = centres.loc[links['src_SA1_7DIG16']][['1', '0']].values
destinations = centres.loc[links['nbr_SA1_7DIG16']][['1', '0']].values


file_location = '2018-06-07-VIC-NeighbourDriveTimes.csv'
    

startTime = time.time()
d = durationGetter(origins, destinations)
d.getDurations(max_i = None)
d.saveDurations(file_location, links[['src_SA1_7DIG16']], links[['nbr_SA1_7DIG16']])
print("{} min taken".format(round((time.time() - startTime)/60, 2)))
