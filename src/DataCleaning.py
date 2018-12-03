from geopy.distance import geodesic
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import requests
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# MORE FEATURES POSSIBLE
# Holidays
# Driving distance (not air distance)
# Driving duration
class DataCleaning(object):

    def __init__(self):
        self.csvColumnsTrain = ['fare_amount', 'pickup_datetime', 'pickup_latitude', 'pickup_longitude',
                                'dropoff_latitude', 'dropoff_longitude', 'passenger_count']

        self.csvColumnsTest = ['pickup_datetime', 'pickup_latitude', 'pickup_longitude',
                               'dropoff_latitude', 'dropoff_longitude', 'passenger_count']

        self.numberOfRows = 5_000_000

        print('**** DATA CLEANING PROCESS ****')
        print()
        print('READING DATA')
        print('-------------------------------------------')
        print('reading training data')
        print('parsing datetime in training data')
        self.dataTrain = pd.read_csv('../input/train.csv', nrows=self.numberOfRows, usecols=self.csvColumnsTrain,
                                     parse_dates=['pickup_datetime'])

        print('reading testing data')
        print('parsing datetime in testing data')
        self.dataTest = pd.read_csv('../input/test.csv', nrows=self.numberOfRows, usecols=self.csvColumnsTest,
                                    parse_dates=['pickup_datetime'])

        print('-------------------------------------------')
        print()

        # Maps and Boundary Box to plot the coordinates on the map for better view of data

        self.nycMap = mpimg.imread('NYCMap.png')
        self.nycMapLandSea = mpimg.imread('NYCMapMask.png')

        self.boundingBoxNYC = (-74.2632, -72.9882, 40.5690, 41.7630)
        self.boundingBoxLaGuardia = (-73.8872, -73.8546, 40.7662, 40.7844)
        self.boundingBoxJFK = (-73.8230, -73.7474, 40.6270, 40.6640)

        self.boundingBoxManhattan = (-74.0190, -73.9263, 40.6968, 40.8668)
        self.boundingBoxBrooklynEast = (-73.9692, -73.8834, 40.5722, 40.7259)
        self.boundingBoxBrooklynWest = (-74.0465, -73.9476, 40.5681, 40.7035)
        self.boundingBoxQueens = (-73.8844, -73.7423, 40.5639, 40.8097)
        self.boundingBoxBronx = (-73.9277, -73.7773, 40.8008, 40.9156)
        self.boundingBoxStatenIsland = (-74.2580, -74.0506, 40.4856, 40.6515)

    def clean(self):
        # Data Cleaning
        # Correcting anomalies or data errors in several of the features.
        # Info: http://home.nyc.gov/html/tlc/html/passenger/taxicab_rate.shtml

        print('Test data size: %d' % len(self.dataTest))
        print()

        print('cleaning training data')

        print('Initial training data size: %d' % len(self.dataTrain))
        print('-------------------------------------------')

        print('removing rows with missing data')
        self.dataTrain = self.dataTrain.dropna(how='any', axis='rows')

        print('removing data located out of NY City')
        self.removeCoordinatesOutNYC()

        print('removing data below minimum fare amount')
        self.dataTrain = self.dataTrain[self.dataTrain['fare_amount'] >= 2.50]

        print('removing data which does not conform the passenger count range (between 1 and 6)')
        self.dataTrain = self.dataTrain[self.dataTrain['passenger_count'].between(1, 6)]

        print('removing data with coordinates in the sea')
        self.removeCoordinatesInWater()

        print('removing data with coordinates at the La Guardia Airport')
        self.removeCoordinatesToFromLaGuardia()

        print('removing data with pickup at the JFK Airport')
        self.removeCoordinatesFromJFKToAny()

        print('setting the pickup and dropoff districts to training data and test data')
        self.addDistricts()

        print('adding distances between pickups and dropoffs to training data and test data')
        self.addDistances()

        print('parsing pickup datetime date to training data and test data')
        self.addPickupParsed()

        print('-------------------------------------------')

        print('New training data size after cleaning: %d' % len(self.dataTrain))
        print()
        print(self.dataTrain.describe())
        print('-------------------------------------------')

        self.dataTrain.to_csv('../input/train_cleaned.csv', index=False)
        print('created a CSV with the training dataset cleaned')

        self.dataTest.to_csv('../input/test_updated.csv', index=False)
        print('updated test dataset with new data')
        print()

    def coordinateToPixel(self, longitude, latitude, dx, dy):
        return (dx * (longitude - self.boundingBoxNYC[0]) / (self.boundingBoxNYC[1] - self.boundingBoxNYC[0])).astype(
            'int'), \
               (dy - dy * (latitude - self.boundingBoxNYC[2]) / (
                       self.boundingBoxNYC[3] - self.boundingBoxNYC[2])).astype('int')

    def removeCoordinatesInWater(self):
        nycLandSea = plt.imread('NYCMapMask.png')[:, :] < 0.9

        rows = nycLandSea.shape[0]  # 553
        colums = nycLandSea.shape[1]  # 468

        # For each coordinate in the training dataset we calculate the pixel position in the NYCLandSea map

        pickup_x, pickup_y = self.coordinateToPixel(self.dataTrain.pickup_longitude, self.dataTrain.pickup_latitude,
                                                    colums, rows)
        dropoff_x, dropoff_y = self.coordinateToPixel(self.dataTrain.dropoff_longitude, self.dataTrain.dropoff_latitude,
                                                      colums, rows)

        # nyc_mask[i, j] -- Matrix with image pixels
        # Boolean indexes reference :
        # 0 is Sea
        # 1 is Land

        index = nycLandSea[pickup_y, pickup_x] & nycLandSea[dropoff_y, dropoff_x]

        # Only coordinates in Land
        self.dataTrain = self.dataTrain[index]

    def isInBoundingBox(self, longitude, latitude, boundingBox):
        if ((boundingBox[0] <= longitude <= boundingBox[1]) &
                (boundingBox[2] <= latitude <= boundingBox[3])):
            return True
        else:
            return False

    def removeCoordinatesOutNYC(self):
        self.dataTrain = self.dataTrain[self.dataTrain['pickup_longitude'].between(self.boundingBoxNYC[0],
                                                                                   self.boundingBoxNYC[1])]
        self.dataTrain = self.dataTrain[self.dataTrain['dropoff_longitude'].between(self.boundingBoxNYC[0],
                                                                                    self.boundingBoxNYC[1])]
        self.dataTrain = self.dataTrain[self.dataTrain['pickup_latitude'].between(self.boundingBoxNYC[2],
                                                                                  self.boundingBoxNYC[3])]
        self.dataTrain = self.dataTrain[self.dataTrain['dropoff_latitude'].between(self.boundingBoxNYC[2],
                                                                                   self.boundingBoxNYC[3])]

    def removeCoordinatesToFromLaGuardia(self):
        # Keeping coordinates where:
        #  NYC West limit < longitude < LG West limit
        #  LG Eastlimit < longitude < NYC East limit
        #  NYC South limit < latitude < LG South limit
        #  LG North  limit < latitude < NYC North limit

        self.dataTrain = self.dataTrain[(self.dataTrain['pickup_longitude'].between(self.boundingBoxNYC[0],
                                                                                    self.boundingBoxLaGuardia[0])
                                         | self.dataTrain['pickup_longitude'].between(self.boundingBoxLaGuardia[1],
                                                                                      self.boundingBoxNYC[1]))

                                        |

                                        (self.dataTrain['pickup_latitude'].between(self.boundingBoxNYC[2],
                                                                                   self.boundingBoxLaGuardia[2])
                                         | self.dataTrain['pickup_latitude'].between(self.boundingBoxLaGuardia[3],
                                                                                     self.boundingBoxNYC[3]))
                                        ]

        self.dataTrain = self.dataTrain[(self.dataTrain['dropoff_longitude'].between(self.boundingBoxNYC[0],
                                                                                     self.boundingBoxLaGuardia[0])
                                         | self.dataTrain['dropoff_longitude'].between(self.boundingBoxLaGuardia[1],
                                                                                       self.boundingBoxNYC[1]))

                                        |

                                        (self.dataTrain['dropoff_latitude'].between(self.boundingBoxNYC[2],
                                                                                    self.boundingBoxLaGuardia[2])
                                         | self.dataTrain['dropoff_latitude'].between(self.boundingBoxLaGuardia[3],
                                                                                      self.boundingBoxNYC[3]))
                                        ]

    def removeCoordinatesFromJFKToAny(self):
        # Keeping coordinates where:
        # NYC West limit < longitude < JFK West limit
        # JFK Eastlimit < longitude < NYC East limit
        # NYC South limit < latitude < JFK South limit
        # JFK North limit < latitude < NYC North limit

        self.dataTrain = self.dataTrain[(self.dataTrain['pickup_longitude'].between(self.boundingBoxNYC[0],
                                                                                    self.boundingBoxJFK[0])
                                         | self.dataTrain['pickup_longitude'].between(self.boundingBoxJFK[1],
                                                                                      self.boundingBoxNYC[1]))

                                        |

                                        (self.dataTrain['pickup_latitude'].between(self.boundingBoxNYC[2],
                                                                                   self.boundingBoxJFK[2])
                                         | self.dataTrain['pickup_latitude'].between(self.boundingBoxJFK[3],
                                                                                     self.boundingBoxNYC[3]))
                                        ]

    def setDistrict(self, longitude, latitude):
        # Districts : 1 - Manhattan / 2 - Brooklyn / 3 - Queens / 4 - Staten Island / 5 - Bronx

        if self.isInBoundingBox(longitude, latitude, self.boundingBoxManhattan):
            return 1

        elif self.isInBoundingBox(longitude, latitude, self.boundingBoxBrooklynEast):
            return 2

        elif self.isInBoundingBox(longitude, latitude, self.boundingBoxBrooklynWest):
            return 2

        elif self.isInBoundingBox(longitude, latitude, self.boundingBoxQueens):
            return 3

        elif self.isInBoundingBox(longitude, latitude, self.boundingBoxStatenIsland):
            return 4

        elif self.isInBoundingBox(longitude, latitude, self.boundingBoxBronx):
            return 5

    def addDistricts(self):
        self.dataTrain['pickup_district'] = self.dataTrain.apply(lambda row:
                                                                 self.setDistrict(row.pickup_longitude,
                                                                                  row.pickup_latitude), axis=1)

        self.dataTrain['dropoff_district'] = self.dataTrain.apply(lambda row:
                                                                  self.setDistrict(row.dropoff_longitude,
                                                                                   row.dropoff_latitude),
                                                                  axis=1)

        self.dataTest['pickup_district'] = self.dataTest.apply(lambda row:
                                                               self.setDistrict(row.pickup_longitude,
                                                                                row.pickup_latitude), axis=1)

        self.dataTest['dropoff_district'] = self.dataTest.apply(lambda row:
                                                                self.setDistrict(row.dropoff_longitude,
                                                                                 row.dropoff_latitude), axis=1)

    # MORE FEATURES POSSIBLE
    def travel(self, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude):
        url = 'https://maps.googleapis.com/maps/api/directions/json?'
        key = '&key=XXXXXX'
        origin = 'origin=' + str(pickup_latitude) + ',' + str(pickup_longitude)
        destination = '&destination=' + str(dropoff_latitude) + ',' + str(dropoff_longitude)

        response = requests.get(url + origin + destination + key)
        distance = response.json()['routes'][0]['legs'][0]['distance']['value']
        duration = response.json()['routes'][0]['legs'][0]['duration']['value']

        return distance, (duration / 60)

    def addDistances(self):
        # self.dataTrain['distance'], self.dataTest['duration'] = self.dataTrain.apply(lambda row:
        #                                                                              self.travel(
        #                                                                                  row.pickup_longitude,
        #                                                                                  row.pickup_latitude,
        #                                                                                  row.dropoff_longitude,
        #                                                                                  row.dropoff_latitude), axis=1)
        #
        # self.dataTest['distance'], self.dataTest['duration'] = self.dataTest.apply(lambda row:
        #                                                                            self.travel(
        #                                                                                row.pickup_longitude,
        #                                                                                row.pickup_latitude,
        #                                                                                row.dropoff_longitude,
        #                                                                                row.dropoff_latitude), axis=1)
        self.dataTrain['distance'] = self.dataTrain.apply(lambda row:
                                                          geodesic((row.pickup_latitude,
                                                                    row.pickup_longitude),
                                                                   (row.dropoff_latitude,
                                                                    row.dropoff_longitude)).miles, axis=1)

        self.dataTest['distance'] = self.dataTest.apply(lambda row:
                                                        geodesic((row.pickup_latitude,
                                                                  row.pickup_longitude),
                                                                 (row.dropoff_latitude,
                                                                  row.dropoff_longitude)).miles, axis=1)

    def addPickupParsed(self):
        self.dataTrain['year'] = self.dataTrain.pickup_datetime.dt.year
        self.dataTrain['month'] = self.dataTrain.pickup_datetime.dt.month
        self.dataTrain['day'] = self.dataTrain.pickup_datetime.dt.day
        self.dataTrain['weekday'] = self.dataTrain.pickup_datetime.dt.weekday
        self.dataTrain['hour'] = self.dataTrain.pickup_datetime.dt.hour

        self.dataTest['year'] = self.dataTest.pickup_datetime.dt.year
        self.dataTest['month'] = self.dataTest.pickup_datetime.dt.month
        self.dataTest['day'] = self.dataTest.pickup_datetime.dt.day
        self.dataTest['weekday'] = self.dataTest.pickup_datetime.dt.weekday
        self.dataTest['hour'] = self.dataTest.pickup_datetime.dt.hour

    def plotOnMap(self, s=1, alpha=0.2):
        print('plotting cleaned training data on map')

        fig, axs = plt.subplots(1, 2, figsize=(16, 10))
        axs[0].scatter(self.dataTrain.pickup_longitude, self.dataTrain.pickup_latitude, zorder=1, alpha=alpha, c='r',
                       s=s)
        axs[0].set_xlim((self.boundingBoxNYC[0], self.boundingBoxNYC[1]))
        axs[0].set_ylim((self.boundingBoxNYC[2], self.boundingBoxNYC[3]))
        axs[0].set_title('Pickup locations')
        axs[0].imshow(self.nycMap, zorder=0, extent=self.boundingBoxNYC)

        axs[1].scatter(self.dataTrain.dropoff_longitude, self.dataTrain.dropoff_latitude, zorder=1, alpha=alpha, c='r',
                       s=s)
        axs[1].set_xlim((self.boundingBoxNYC[0], self.boundingBoxNYC[1]))
        axs[1].set_ylim((self.boundingBoxNYC[2], self.boundingBoxNYC[3]))
        axs[1].set_title('Dropoff locations')
        axs[1].imshow(self.nycMap, zorder=0, extent=self.boundingBoxNYC)

        plt.show()
