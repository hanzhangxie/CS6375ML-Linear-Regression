#####################################################################################################################
#   CS 6375.003 - Assignment 1, Linear Regression using Gradient Descent
#   This is a simple starter code in Python 3.6 for linear regression using the notation shown in class.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   test - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class LinearRegression:
    def __init__(self, train):
        np.random.seed(1)
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        df = pd.read_csv(train,index_col=0)
        df.insert(0, 'X0', 1)
        self.nrows, self.ncols = df.shape[0], df.shape[1]
        self.X =  df.iloc[:, 0:(self.ncols -1)].values.reshape(self.nrows, self.ncols-1)
        self.y = df.iloc[:, (self.ncols-1)].values.reshape(self.nrows, 1)
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)

    # TODO: Perform pre-processing for your dataset. It may include doing the following:
    #   - getting rid of null values
    #   - converting categorical to numerical values
    #   - scaling and standardizing attributes
    #   - anything else that you think could increase model performance
    # Below is the pre-process function
    def preProcess(self):
        data = pd.read_csv("Automobile Dataset.csv")
        # remove null or NA values
        data = data.dropna()
        # remove any redundant rows
        data = data.drop_duplicates()
        # get rid of attributes that are not correlated with the outcome
        data = data.drop(columns=['symboling','normalized_losses','aspiration','engine_location','wheel_base','length','width',
        'height','curb_weight','engine_type','engine_size','fuel_system','bore','stroke','compression_ratio','peak_rpm'])
        # convert categorical variables to numerical variables
        data1 = pd.DataFrame(data, columns=['make','fuel_type','num_of_doors','body_style','drive_wheels','num_of_cylinders','horsepower','city_mpg','highway_mpg','price'])
        make_mapping = {'alfa-romero': 0, 'audi': 1, 'bmw': 2, 'chevrolet': 3, 'dodge': 4, 'honda': 5,
                    'isuzu': 6, 'jaguar': 7, 'mazda': 8, 'mercedes-benz': 9, 'mercury': 10,
                    'mitsubishi': 11, 'nissan': 12, 'peugot': 13, 'plymouth': 14, 'porsche': 15, 
                    'renault': 16, 'saab': 17, 'subaru': 18, 'toyota': 19, 'volkswagen': 20, 
                    'volvo': 21}
        fuel_type_mapping = {'diesel': 0, 'gas': 1}
        num_of_doors_mapping = {'four': 0, 'two': 1}
        body_style_mapping = {'hardtop': 0,'sedan': 1, 'hatchback': 2, 'convertible': 3, 'wagon':4}
        drive_wheels_mapping = {'4wd': 0, 'fwd': 1, 'rwd': 2}
        num_of_cylinders_mapping = {'eight': 8, 'five': 5, 'four': 4, 'six': 6, 'three': 3, 'twelve': 12, 
                                'two': 2}
        data1['make'] = data1['make'].map(make_mapping)
        data1['fuel_type'] = data1['fuel_type'].map(fuel_type_mapping)
        data1['num_of_doors'] = data1['num_of_doors'].map(num_of_doors_mapping)
        data1['body_style'] = data1['body_style'].map(body_style_mapping)
        data1['drive_wheels'] = data1['drive_wheels'].map(drive_wheels_mapping)
        data1['num_of_cylinders'] = data1['num_of_cylinders'].map(num_of_cylinders_mapping)
        
        data1['price']=data1.apply(lambda x: x['price']*0.0001, axis=1)
        # random sort the dataset
        data1 = shuffle(data1)
        # split the dataset into 80:20 to create train and test datasets
        length=len(data1)
        testlen=int(length*0.2)
        test1=data1[0:testlen]
        train1=data1[testlen:length]
        pd.DataFrame.to_csv(test1,"test.csv")
        pd.DataFrame.to_csv(train1,"train.csv")

    # Below is the training function
    def train(self, epochs = 500, learning_rate = 0.00015):
        # Perform Gradient Descent
        for i in range(epochs):
            # Make prediction with current weights
            h = np.dot(self.X, self.W)
            # Find error
            error = h - self.y
            self.W = self.W - (1 / self.nrows) * learning_rate * np.dot(self.X.T, error)

        return self.W, error

    # predict on test dataset
    def predict(self, test):
        testDF = pd.read_csv(test,index_col=0)
        testDF.insert(0, "X0", 1)
        nrows, ncols = testDF.shape[0], testDF.shape[1]
        testX = testDF.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        testY = testDF.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        pred = np.dot(testX, self.W)
        error = pred - testY
        mse = 1/(2*nrows) * np.dot(error.T, error)
        return mse

if __name__ == "__main__":
    model1 = LinearRegression("Automobile Dataset.csv")
    model1.preProcess()
    model = LinearRegression("train.csv")
    W, e = model.train()
    mse = model.predict("test.csv")
    print(mse)