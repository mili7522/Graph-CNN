import geopandas as gpd
import pandas as pd

# Load shapefiles for Australia (SA1)
austSA1 = gpd.read_file('Geography/1270055001_sa1_2016_aust_shape/SA1_2016_AUST.shp')


vicSA1 = austSA1[austSA1['GCC_NAME16'].isin(['Rest of Vic.', 'Greater Melbourne'])]  # Exclude some cases of 'Migratory - Offshore - Shipping (NSW)'


# Save centroids
centroids_x = vicSA1.centroid.x
centroids_y = vicSA1.centroid.y

centroids = pd.DataFrame([centroids_y, centroids_x])
centroids = centroids.T


### Save the filtered shapefile
vicSA1.to_file('Geography/2018-06-07-VIC2016SA1')


centroids.index = vicSA1['SA1_7DIG16']
centroids.to_csv('Geography/2018-06-07-VIC-SA1-2016Centres.csv', index = True)
