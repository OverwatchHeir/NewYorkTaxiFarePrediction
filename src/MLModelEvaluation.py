import numpy
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class MLModelEvaluation(object):

    def __init__(self):
        self.csvColumnsTrain = ['fare_amount', 'pickup_latitude', 'pickup_longitude',
                                'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'pickup_district',
                                'dropoff_district', 'distance', 'year',
                                'month', 'day', 'weekday', 'hour']

        self.csvColumnsTest = ['pickup_latitude', 'pickup_longitude',
                               'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'pickup_district',
                               'dropoff_district', 'distance', 'year',
                               'month', 'day', 'weekday', 'hour']

        self.numberOfRows = 5_000_000

        print('**** EVALUATING MACHINE LEARNING MODELS ****')
        print()
        print('READING CLEANED DATA')
        print('-------------------------------------------')
        print('reading cleaned training data')
        self.dataTrain = pd.read_csv('../input/train_cleaned.csv', nrows=self.numberOfRows,
                                     usecols=self.csvColumnsTrain)

        print('reading testing data')
        self.dataTest = pd.read_csv('../input/test_updated.csv', nrows=self.numberOfRows, usecols=self.csvColumnsTest)

        self.dataTrain = self.dataTrain.dropna(how='any', axis='rows')
        self.X = self.dataTrain.drop('fare_amount', axis=1)
        self.y = self.dataTrain['fare_amount']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25,
                                                                                random_state=123)

    def mlModelEvaluation(self):
        # Data preparation (dividing the training data into train and validation data sets)
        # It allows us to evaluate machine learning models and helps us to understand how the model fits using
        # training data , and works on any unseen data.

        # This way, models can be categorized in OVERFITTING or UNDERFITTING
        # Huge variation in the training and validation RMSE  indicates overfitting
        # We use the RMSE, bias and variance to see the model suitability
        # Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors)
        # Bias here is a source of error in your model that causes it to over-generalize and underfit your data.
        # In contrast, variance is sensitivity to noise in the data that causes your model to overfit.

        # A good model has low bias and variance (to avoid overfitting)!!

        # Linear Regression Model
        # linearRegressionTestRMSE, linearRegressionTrainRMSE, \
        #     linearRegressionVariance = self.linearRegressionEvaluation()
        #
        # print("Train RMSE for Linear Regression :", linearRegressionTrainRMSE)
        # print("Test RMSE for Linear Regression : ", linearRegressionTestRMSE)
        # print("Variance for Linear Regression :", linearRegressionVariance)

        # Ramdom Forest Model
        # randomForestTestRMSE, randomForestTrainRMSE, randomForestVariance = self.randomForestEvaluation()
        #
        # print("Train RMSE for Random Forest : ", randomForestTrainRMSE)
        # print("Test RMSE for Random Forest : ", randomForestTestRMSE)
        # print("Variance for Random Forest : ", randomForestVariance)

        # Gradient Boosting tree based algorithm (LightGBM)
        # grows tree vertically while other algorithm grows trees horizontally
        # lgbTreeTestRMSE, lgbTrainRMSE, lgbTreeVariance = self.lgbTreeEvaluation()
        #
        # print("Train RMSE for Light GBM :", lgbTrainRMSE)
        # print("Test RMSE for Light GBM :", lgbTreeTestRMSE)
        # print("Variance for Light GBM : ", lgbTreeVariance)

        # # Neural Network
        neuralNetworkTestRMSE, neuralNetworkTrainRMSE, neuralNetworkVariance = self.neuralNetworkEvaluation()
        print("Train RMSE for Neural Network  :", neuralNetworkTrainRMSE)
        print("Test RMSE for Neural Network  :", neuralNetworkTestRMSE)
        print("Variance for Neural Network : ", neuralNetworkVariance)

        # self.generateSubmission()

    def compute_rmse(self, predicted, actual):
        return np.sqrt(mean_squared_error(predicted, actual))

    def compute_variance(self, train_rmse, test_rmse):
        return abs(train_rmse - test_rmse)

    def linearRegressionEvaluation(self):
        print('-------------------------------------------')
        print('Evaluating a Linear Regression Model')

        linearRegression = LinearRegression()
        model = linearRegression.fit(self.X_train, self.y_train)
        predictionTest = linearRegression.predict(self.X_test)
        predictionTrain = linearRegression.predict(self.X_train)

        print(model)

        linearRegressionTestRMSE = self.compute_rmse(predictionTest, self.y_test)
        linearRegressionTrainRMSE = self.compute_rmse(predictionTrain, self.y_train)
        linearRegressionVariance = self.compute_variance(linearRegressionTrainRMSE, linearRegressionTestRMSE)

        return linearRegressionTestRMSE, linearRegressionTrainRMSE, linearRegressionVariance

    def randomForestEvaluation(self):
        print('-------------------------------------------')
        print('Evaluating a Random Forest Model')
        randomForest = RandomForestRegressor(n_estimators=100, random_state=123, n_jobs=-1, verbose=1)
        model = randomForest.fit(self.X_train, self.y_train)
        predictionTest = randomForest.predict(self.X_test)
        predictionTrain = randomForest.predict(self.X_train)

        print(model)

        randomForestTrainRMSE = self.compute_rmse(predictionTrain, self.y_train)
        randomForestTestRMSE = self.compute_rmse(predictionTest, self.y_test)
        randomForestVariance = self.compute_variance(randomForestTrainRMSE, randomForestTestRMSE)

        return randomForestTestRMSE, randomForestTrainRMSE, randomForestVariance

    def lgbTreeEvaluation(self):
        print('-------------------------------------------')
        print('Evaluating a Light Gradient Boosting Tree Model')
        train_data = lgb.Dataset(self.X_train, label=self.y_train, silent=True)

        # Tuning LightGBM Tree
        parameters = {
                      'learning_rate': 0.01,
                      'num_leaves': 100,  # 2^(max_depth) <= num_leaves
                      'max_depth': 20,  # to handle model overfitting
                      'num_trees': 5000,
                      'application': 'regression',
                      'verbosity': -1,
                      'metric': 'RMSE'
                      }

        results = lgb.cv(parameters, train_data, num_boost_round=500, nfold=5, verbose_eval=20,
                         early_stopping_rounds=20, stratified=False)

        lgbBoost = lgb.train(parameters, train_data, len(results['rmse-mean']))
        predictionTest = lgbBoost.predict(self.X_test, num_iteration=lgbBoost.best_iteration)
        predictionTrain = lgbBoost.predict(self.X_train, num_iteration=lgbBoost.best_iteration)

        lgbTrainRMSE = self.compute_rmse(predictionTrain, self.y_train)
        lgbTreeTestRMSE = self.compute_rmse(predictionTest, self.y_test)
        lgbTreeVariance = self.compute_variance(lgbTrainRMSE, lgbTreeTestRMSE)

        return lgbTreeTestRMSE, lgbTrainRMSE, lgbTreeVariance

    def neuralNetworkEvaluation(self):
        print('-------------------------------------------')
        print('Evaluating a Neural Network Model')

        def nn_model():
            model = Sequential()
            # input layer
            # columns except pickup_datetime

            # hidden layer
            model.add(Dense(13, input_dim=13, activation='relu'))

            # output layer
            model.add(Dense(1, activation='linear'))

            # ADAM optimization algorithm is used and a mean squared error loss function is optimized
            model.compile(loss='mean_squared_error', optimizer='adam')

            return model

        estimator = KerasRegressor(build_fn=nn_model, nb_epoch=100, batch_size=5, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        estimator.fit(self.X_train, self.y_train, epochs=100, verbose=1, callbacks=[early_stop])

        predictionTrain = estimator.predict(self.X_train)
        predictionTest = estimator.predict(self.X_test)

        neuralNetworkTrainRMSE = self.compute_rmse(predictionTrain, self.y_train)
        neuralNetworkTestRMSE = self.compute_rmse(predictionTest, self.y_test)
        neuralNetworkVariance = self.compute_variance(neuralNetworkTrainRMSE, neuralNetworkTestRMSE)

        return neuralNetworkTestRMSE, neuralNetworkTrainRMSE, neuralNetworkVariance

    def generateSubmission(self):
        # We use the algorithm with the lowest RMSE to predict the fare amount of the ride
        X_test = self.dataTest.drop(['pickup_datetime'], axis=1)
        randomForest = RandomForestRegressor(n_estimators=150, n_jobs=-1)
        prediction = randomForest.predict(X_test)

        submission = pd.DataFrame({'key': self.dataTest.key,
                                   'fare_amount': prediction,
                                   'distance': self.dataTest.distance,
                                   'pickup_latitude': self.dataTest.pickup_latitude,
                                   'pickup_longitude': self.dataTest.pickup_longitude,
                                   'dropoff_latitude': self.dataTest.dropoff_latitude,
                                   'dropoff_longitude': self.dataTest.dropoff_longitude
                                   }, columns=['key',
                                               'fare_amount',
                                               'distance',
                                               'pickup_latitude',
                                               'pickup_longitude',
                                               'dropoff_latitude',
                                               'dropoff_longitude'
                                               ])
        submission.to_csv('../input/results.csv', index=False)
