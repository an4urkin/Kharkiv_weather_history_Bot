import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error


def meantemp(data_frame, test_frame):

    print(data_frame.describe().T)

    print(data_frame.corr()[['meantemp']].sort_values('meantemp'))

    predictors = ['meantemp_1', 'meantemp_2', 'meantemp_3',
                  'maxtemp_1', 'maxtemp_2', 'maxtemp_3',
                  'mintemp_1', 'mintemp_2', 'mintemp_3',
                  'meandewpt_1', 'meandewpt_2', 'meandewpt_3',
                  'maxdewpt_1', 'maxdewpt_2', 'maxdewpt_3',
                  'mindewpt_1', 'mindewpt_2', 'mindewpt_3']
    df2 = data_frame[['meantemp'] + predictors]
    df_test2 = test_frame[predictors]

    # manually set the parameters of the figure to and appropriate size
    plt.rcParams['figure.figsize'] = [16, 22]
    
    # call subplots specifying the grid structure we desire and that
    # the y axes should be shared
    fig, axes = plt.subplots(nrows=6, ncols=3, sharey=True)
    
    # Since it would be nice to loop through the features in to build this plot
    # let us rearrange our data into a 2D array of 6 rows and 3 columns
    arr = np.array(predictors).reshape(6, 3)
    
    # use enumerate to loop over the arr 2D array of rows and columns
    # and create scatter plots of each meantempm vs each feature
    for row, col_arr in enumerate(arr):
        for col, feature in enumerate(col_arr):
            axes[row, col].scatter(df2[feature], df2['meantemp'])
            if col == 0:
                axes[row, col].set(xlabel=feature, ylabel='meantemp')
            else:
                axes[row, col].set(xlabel=feature)
    plt.show()

    # separate our my predictor variables (X) from my outcome variable y
    x = df2[predictors]
    y = df2['meantemp']

    # Add a constant to the predictor variable set to represent the Bo intercept
    x = sm.add_constant(x)
    # print(x.iloc[:5, :5])

    # (1) select a significance value
    alpha = 0.02
    # model = sm.OLS(y, x).fit()
    #
    # print(model.summary())

    x = x.drop('meantemp_1', axis=1)
    x = x.drop('meantemp_2', axis=1)
    x = x.drop('meantemp_3', axis=1)
    x = x.drop('maxtemp_2', axis=1)
    x = x.drop('maxtemp_3', axis=1)
    x = x.drop('mintemp_2', axis=1)
    x = x.drop('meandewpt_2', axis=1)
    x = x.drop('meandewpt_3', axis=1)
    x = x.drop('maxdewpt_1', axis=1)
    x = x.drop('maxdewpt_2', axis=1)
    # my_model = sm.OLS(y, x).fit()
    model = sm.OLS(y, x).fit()  # fit_my_meantemp(x, y)

    # print(model.summary())

    to_remove = ['meantemp_1', 'meantemp_2', 'meantemp_3', 'maxtemp_2', 'maxtemp_3', 'mintemp_2', 'meandewpt_2',
                 'meandewpt_3', 'maxdewpt_1', 'maxdewpt_2']
    to_keep = [col for col in df_test2 if col not in to_remove]
    df_test2 = df_test2[to_keep]

    x = x.drop('const', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    # instantiate the regressor class
    regressor = LinearRegression()

    # fit the build the model by fitting the regressor to the training data
    regressor.fit(x_train, y_train)

    # make a prediction set using the test set
    prediction = regressor.predict(x_test)

    # Evaluate the prediction accuracy of the model
    print("The Explained Variance: %.2f" % regressor.score(x_test, y_test))
    print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
    print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))

    return regressor


def maxtemp(data_frame, test_frame):
    print(data_frame.describe().T)

    print(data_frame.corr()[['maxtemp']].sort_values('maxtemp'))

    predictors = ['meantemp_1', 'meantemp_2', 'meantemp_3',
                  'maxtemp_1', 'maxtemp_2', 'maxtemp_3',
                  'mintemp_1', 'mintemp_2', 'mintemp_3',
                  'meandewpt_1', 'meandewpt_2', 'meandewpt_3',
                  'maxdewpt_1', 'maxdewpt_2', 'maxdewpt_3',
                  'mindewpt_1', 'mindewpt_2', 'mindewpt_3',
                  'meanhumidity_1', 'meanhumidity_2', 'meanhumidity_3',
                  'minhumidity_1', 'minhumidity_2', 'minhumidity_3'
                  ]
    df2 = data_frame[['maxtemp'] + predictors]
    df_test2 = test_frame[predictors]

    # manually set the parameters of the figure to and appropriate size
    plt.rcParams['figure.figsize'] = [16, 22]
    
    # call subplots specifying the grid structure we desire and that
    # the y axes should be shared
    fig, axes = plt.subplots(nrows=8, ncols=3, sharey=True)
    
    # Since it would be nice to loop through the features in to build this plot
    # let us rearrange our data into a 2D array of 6 rows and 3 columns
    arr = np.array(predictors).reshape(8, 3)
    
    # use enumerate to loop over the arr 2D array of rows and columns
    # and create scatter plots of each meantempm vs each feature
    for row, col_arr in enumerate(arr):
        for col, feature in enumerate(col_arr):
            axes[row, col].scatter(df2[feature], df2['maxtemp'])
            if col == 0:
                axes[row, col].set(xlabel=feature, ylabel='maxtemp')
            else:
                axes[row, col].set(xlabel=feature)
    plt.show()

    # separate our my predictor variables (X) from my outcome variable y
    x = df2[predictors]
    y = df2['maxtemp']

    # Add a constant to the predictor variable set to represent the Bo intercept
    x = sm.add_constant(x)
    print(x.iloc[:5, :5])

    # (1) select a significance value
    alpha = 0.02

    model = sm.OLS(y, x).fit()  # fit_my_model(x, y)
    # print(model.summary())

    x = x.drop('meantemp_1', axis=1)
    x = x.drop('meantemp_2', axis=1)
    x = x.drop('meantemp_3', axis=1)
    x = x.drop('maxtemp_2', axis=1)
    x = x.drop('maxtemp_3', axis=1)
    x = x.drop('mintemp_2', axis=1)
    x = x.drop('mintemp_1', axis=1)
    x = x.drop('meandewpt_2', axis=1)
    x = x.drop('meandewpt_3', axis=1)
    x = x.drop('maxdewpt_1', axis=1)
    x = x.drop('maxdewpt_2', axis=1)
    x = x.drop('mindewpt_2', axis=1)
    x = x.drop('meanhumidity_1', axis=1)
    x = x.drop('meanhumidity_3', axis=1)
    x = x.drop('minhumidity_1', axis=1)
    x = x.drop('minhumidity_2', axis=1)
    x = x.drop('minhumidity_3', axis=1)
    model = sm.OLS(y, x).fit()  # fit_my_model(x, y)

    to_remove = ['meantemp_1', 'meantemp_2', 'meantemp_3', 'maxtemp_2', 'maxtemp_3', 'mintemp_2', 'mintemp_1',
                 'meandewpt_2', 'meandewpt_3', 'maxdewpt_1', 'maxdewpt_2', 'mindewpt_2', 'meanhumidity_1',
                 'meanhumidity_3', 'minhumidity_1', 'minhumidity_2', 'minhumidity_3']
    to_keep = [col for col in df_test2 if col not in to_remove]
    df_test2 = df_test2[to_keep]

    # print(model.summary())

    x = x.drop('const', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    # instantiate the regressor class
    regressor = LinearRegression()

    # fit the build the model by fitting the regressor to the training data
    regressor.fit(x_train, y_train)

    # make a prediction set using the test set
    prediction = regressor.predict(x_test)
    print(prediction)

    # Evaluate the prediction accuracy of the model
    print("The Explained Variance: %.2f" % regressor.score(x_test, y_test))
    print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
    print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))

    return regressor


def mintemp(data_frame, test_frame):
    print(data_frame.describe().T)

    print(data_frame.corr()[['mintemp']].sort_values('mintemp'))

    predictors = ['meantemp_1', 'meantemp_2', 'meantemp_3',
                  'maxtemp_1', 'maxtemp_2', 'maxtemp_3',
                  'mintemp_1', 'mintemp_2', 'mintemp_3',
                  'meandewpt_1', 'meandewpt_2', 'meandewpt_3',
                  'maxdewpt_1', 'maxdewpt_2', 'maxdewpt_3',
                  'mindewpt_1', 'mindewpt_2', 'mindewpt_3',
                  'meanhumidity_2', 'meanhumidity_3',
                  'minhumidity_1', 'minhumidity_2', 'minhumidity_3'
                  ]
    df2 = data_frame[['mintemp'] + predictors]
    df_test2 = test_frame[predictors]

    # manually set the parameters of the figure to and appropriate size
    plt.rcParams['figure.figsize'] = [16, 22]
    
    # call subplots specifying the grid structure we desire and that
    # the y axes should be shared
    fig, axes = plt.subplots(nrows=8, ncols=3, sharey=True)
    
    # Since it would be nice to loop through the features in to build this plot
    # let us rearrange our data into a 2D array of 6 rows and 3 columns
    arr = np.array(predictors).reshape(8, 3)
    
    # use enumerate to loop over the arr 2D array of rows and columns
    # and create scatter plots of each meantempm vs each feature
    for row, col_arr in enumerate(arr):
        for col, feature in enumerate(col_arr):
            axes[row, col].scatter(df2[feature], df2['mintemp'])
            if col == 0:
                axes[row, col].set(xlabel=feature, ylabel='mintemp')
            else:
                axes[row, col].set(xlabel=feature)
    plt.show()

    # separate our my predictor variables (X) from my outcome variable y
    x = df2[predictors]
    y = df2['mintemp']

    # Add a constant to the predictor variable set to represent the Bo intercept
    x = sm.add_constant(x)
    # print(x.iloc[:5, :5])

    # (1) select a significance value
    alpha = 0.02

    model = sm.OLS(y, x).fit()  # fit_my_model(x, y)
    # print(model.summary())

    x = x.drop('meantemp_1', axis=1)
    x = x.drop('meantemp_2', axis=1)
    x = x.drop('meantemp_3', axis=1)
    x = x.drop('maxtemp_3', axis=1)
    x = x.drop('mintemp_2', axis=1)
    x = x.drop('meandewpt_1', axis=1)
    x = x.drop('meandewpt_2', axis=1)
    x = x.drop('meandewpt_3', axis=1)
    x = x.drop('maxdewpt_1', axis=1)
    x = x.drop('maxdewpt_3', axis=1)
    x = x.drop('maxdewpt_2', axis=1)
    x = x.drop('mindewpt_3', axis=1)
    x = x.drop('meanhumidity_2', axis=1)
    x = x.drop('meanhumidity_3', axis=1)
    x = x.drop('minhumidity_2', axis=1)
    x = x.drop('minhumidity_3', axis=1)
    model = sm.OLS(y, x).fit()  # fit_my_model(x, y)

    print(model.summary())

    to_remove = ['meantemp_1', 'meantemp_2', 'meantemp_3', 'maxtemp_3', 'mintemp_2', 'meandewpt_2', 'meandewpt_3',
                 'meandewpt_1', 'maxdewpt_1', 'maxdewpt_2', 'maxdewpt_3', 'mindewpt_3', 'meanhumidity_2',
                 'meanhumidity_3', 'minhumidity_2', 'minhumidity_3']
    to_keep = [col for col in df_test2 if col not in to_remove]
    df_test2 = df_test2[to_keep]

    x = x.drop('const', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    # instantiate the regressor class
    regressor = LinearRegression()

    # fit the build the model by fitting the regressor to the training data
    regressor.fit(x_train, y_train)

    # make a prediction set using the test set
    prediction = regressor.predict(x_test)#df_test2)
    print(prediction)

    Evaluate the prediction accuracy of the model
    print("The Explained Variance: %.2f" % regressor.score(x_test, y_test))
    print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
    print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))

    return regressor


def meanhumidity(data_frame, test_frame):
    print(data_frame.describe().T)

    print(data_frame.corr()[['meanhumidity']].sort_values('meanhumidity'))

    predictors = ['meantemp_1', 'meantemp_2', 'meantemp_3',
                  'maxtemp_1', 'maxtemp_2', 'maxtemp_3',
                  'mintemp_2', 'maxhumidity_2', 'maxhumidity_1',
                  'meanhumidity_3', 'minhumidity_3', 'minhumidity_2',
                  'meanhumidity_2', 'minhumidity_1', 'meanhumidity_1'
                  ]
    df2 = data_frame[['meanhumidity'] + predictors]
    df_test2 = test_frame[predictors]

    # manually set the parameters of the figure to and appropriate size
    plt.rcParams['figure.figsize'] = [16, 22]
    
    # call subplots specifying the grid structure we desire and that
    # the y axes should be shared
    fig, axes = plt.subplots(nrows=5, ncols=3, sharey=True)
    
    # Since it would be nice to loop through the features in to build this plot
    # let us rearrange our data into a 2D array of 6 rows and 3 columns
    arr = np.array(predictors).reshape(5, 3)
    
    # use enumerate to loop over the arr 2D array of rows and columns
    # and create scatter plots of each meantempm vs each feature
    for row, col_arr in enumerate(arr):
        for col, feature in enumerate(col_arr):
            axes[row, col].scatter(df2[feature], df2['meanhumidity'])
            if col == 0:
                axes[row, col].set(xlabel=feature, ylabel='meanhumidity')
            else:
                axes[row, col].set(xlabel=feature)
    plt.show()

    # separate our my predictor variables (X) from my outcome variable y
    x = df2[predictors]
    y = df2['meanhumidity']

    # Add a constant to the predictor variable set to represent the Bo intercept
    x = sm.add_constant(x)
    # print(x.iloc[:5, :5])

    # (1) select a significance value
    alpha = 0.02

    model = sm.OLS(y, x).fit()  # fit_my_model(x, y)
    print(model.summary())

    x = x.drop('meantemp_1', axis=1)
    x = x.drop('meantemp_2', axis=1)
    x = x.drop('meantemp_3', axis=1)
    x = x.drop('maxtemp_2', axis=1)
    x = x.drop('maxtemp_3', axis=1)
    x = x.drop('mintemp_2', axis=1)
    x = x.drop('meanhumidity_3', axis=1)
    x = x.drop('minhumidity_2', axis=1)
    x = x.drop('minhumidity_1', axis=1)
    x = x.drop('maxhumidity_2', axis=1)
    x = x.drop('maxhumidity_1', axis=1)
    model = sm.OLS(y, x).fit()  # fit_my_model(x, y)

    print(model.summary())

    to_remove = ['meantemp_1', 'meantemp_2', 'meantemp_3', 'maxtemp_3', 'maxtemp_2', 'mintemp_2',
                 'meanhumidity_3', 'minhumidity_2', 'minhumidity_1', 'maxhumidity_2', 'maxhumidity_1']
    to_keep = [col for col in df_test2 if col not in to_remove]
    df_test2 = df_test2[to_keep]

    x = x.drop('const', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    # instantiate the regressor class
    regressor = LinearRegression()

    # fit the build the model by fitting the regressor to the training data
    regressor.fit(x_train, y_train)

    # make a prediction set using the test set
    prediction = regressor.predict(x_test)
    print(prediction)

    Evaluate the prediction accuracy of the model
    print("The Explained Variance: %.2f" % regressor.score(x_test, y_test))
    print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
    print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))

    return regressor


def clear_mean(df):
    predictors = ['meantemp_1', 'meantemp_2', 'meantemp_3',
                  'maxtemp_1', 'maxtemp_2', 'maxtemp_3',
                  'mintemp_1', 'mintemp_2', 'mintemp_3',
                  'meandewpt_1', 'meandewpt_2', 'meandewpt_3',
                  'maxdewpt_1', 'maxdewpt_2', 'maxdewpt_3',
                  'mindewpt_1', 'mindewpt_2', 'mindewpt_3']
    dataf = df[predictors]
    to_remove = ['meantemp_1', 'meantemp_2', 'meantemp_3', 'maxtemp_2', 'maxtemp_3', 'mintemp_2', 'meandewpt_2',
                 'meandewpt_3', 'maxdewpt_1', 'maxdewpt_2']
    to_keep = [col for col in dataf if col not in to_remove]
    dataf = dataf[to_keep]
    return dataf


def clear_max(df):
    predictors = ['meantemp_1', 'meantemp_2', 'meantemp_3',
                  'maxtemp_1', 'maxtemp_2', 'maxtemp_3',
                  'mintemp_1', 'mintemp_2', 'mintemp_3',
                  'meandewpt_1', 'meandewpt_2', 'meandewpt_3',
                  'maxdewpt_1', 'maxdewpt_2', 'maxdewpt_3',
                  'mindewpt_1', 'mindewpt_2', 'mindewpt_3',
                  'meanhumidity_1', 'meanhumidity_2', 'meanhumidity_3',
                  'minhumidity_1', 'minhumidity_2', 'minhumidity_3'
                  ]
    dataf = df[predictors]
    to_remove = ['meantemp_1', 'meantemp_2', 'meantemp_3', 'maxtemp_2', 'maxtemp_3', 'mintemp_2', 'mintemp_1',
                 'meandewpt_2', 'meandewpt_3', 'maxdewpt_1', 'maxdewpt_2', 'mindewpt_2', 'meanhumidity_1',
                 'meanhumidity_3', 'minhumidity_1', 'minhumidity_2', 'minhumidity_3']
    to_keep = [col for col in dataf if col not in to_remove]
    dataf = dataf[to_keep]
    return dataf


def clear_min(df):
    predictors = ['meantemp_1', 'meantemp_2', 'meantemp_3',
                  'maxtemp_1', 'maxtemp_2', 'maxtemp_3',
                  'mintemp_1', 'mintemp_2', 'mintemp_3',
                  'meandewpt_1', 'meandewpt_2', 'meandewpt_3',
                  'maxdewpt_1', 'maxdewpt_2', 'maxdewpt_3',
                  'mindewpt_1', 'mindewpt_2', 'mindewpt_3',
                  'meanhumidity_2', 'meanhumidity_3',
                  'minhumidity_1', 'minhumidity_2', 'minhumidity_3'
                  ]
    dataf = df[predictors]
    to_remove = ['meantemp_1', 'meantemp_2', 'meantemp_3', 'maxtemp_3', 'mintemp_2', 'meandewpt_2', 'meandewpt_3',
                 'meandewpt_1', 'maxdewpt_1', 'maxdewpt_2', 'maxdewpt_3', 'mindewpt_3', 'meanhumidity_2',
                 'meanhumidity_3', 'minhumidity_2', 'minhumidity_3']
    to_keep = [col for col in dataf if col not in to_remove]
    dataf = dataf[to_keep]
    return dataf


def clear_humidity(df):
    predictors = ['meantemp_1', 'meantemp_2', 'meantemp_3',
                  'maxtemp_1', 'maxtemp_2', 'maxtemp_3',
                  'mintemp_2', 'maxhumidity_2', 'maxhumidity_1',
                  'meanhumidity_3', 'minhumidity_3', 'minhumidity_2',
                  'meanhumidity_2', 'minhumidity_1', 'meanhumidity_1'
                  ]
    dataf = df[predictors]
    to_remove = ['meantemp_1', 'meantemp_2', 'meantemp_3', 'maxtemp_3', 'maxtemp_2', 'mintemp_2',
                 'meanhumidity_3', 'minhumidity_2', 'minhumidity_1', 'maxhumidity_2', 'maxhumidity_1']
    to_keep = [col for col in dataf if col not in to_remove]
    dataf = dataf[to_keep]
    return dataf


# READ DATAFRAME FROM PICKLED FILE
with open(os.path.join('./sorted_data_df_test.pickle'), 'rb') as f:
    df = pickle.load(f)

df_test = pd.read_csv('test_may.csv').set_index('date')

# CREATE MODELS
meanhumidity_model = meanhumidity(df, df_test)
meantemp_model = meantemp(df, df_test)
maxtemp_model = maxtemp(df, df_test)
mintemp_model = mintemp(df, df_test)

# # STORE MODELS IN PICKLED FILES
with open(os.path.join('./meantemp_model.pickle'), 'wb') as f:
    pickle.dump(meantemp_model, f)
with open(os.path.join('./maxtemp_model.pickle'), 'wb') as f:
    pickle.dump(maxtemp_model, f)
with open(os.path.join('./mintemp_model.pickle'), 'wb') as f:
    pickle.dump(mintemp_model, f)
with open(os.path.join('./meanhumidity_model.pickle'), 'wb') as f:
    pickle.dump(meanhumidity_model, f)

