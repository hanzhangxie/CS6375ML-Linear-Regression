# Linear-Regression

Linear Regression using Gradient Descent and using ML libraries

Part 1 - Using Gradient Descent
===============================
1. This dataset comes from 1985 Auto Imports Database. All instances with missing attribute values are discarded and all categorical variables are converted into numerical values. And some of attributes that are not closely correlated with the outcome are dropped in preProcess method. The goal of learning is to predict price of car using all numeric and Boolean attributes.

	The final attributes --- make, fuel_type, num_of_doors, body_style, drive_wheels, num_of_cylinders, horsepower, city_mpg, highway_mpg (9)

	The labels --- price (1)

2. Development environment: Python 3.7.2 

	Package used: numpy, pandas, sklearn 

	Method: Linear Regression using Gradient Descent

3. Using 'shuffle' method to random sort the dataset and splitting it into 80:20 to create train and test datasets. This resulted with a training set of 128 instances and a testing set of 31 instances.

4. Results: 
	
	Mean Squared Error(MSE) of Prediction from Actual --- 0.11696754
            
	The value of learning rate --- 0.00015
            
	The number of epochs parameter --- 500

5. It looks like that the MSE result is good but we are not satisfied with the value of epochs and learning rate. We have tried values of epoch from 1 to 10000, and the corresponding MSE falls very quickly at first(when epoch is from 1 to 10), and then it gradually decreases at a much slower speed. The learning rate is also relatively small. We think the reason may be that the amount of dataset is insufficient to learn while there are so many attributes relating to the outputs.

Part 2 - Using ML libraries
=============================================
1. Development environment: Python 3.7.2
   
   Package used: numpy, matplotlib, scikit learn’s linear regression
   
   Method: Linear Regression using ML libraries

2. Firstly, splitting the dataset into training and testing sets. Then using the LinearRegression package to fit a model based on the training set. And make predictions using the testing set to see if the test data is matched with the linear regression model. At last, calculating weight coefficients, MSE and R2 value for some important attributes and outputting plots that show the regression relationship between the attribute and the price.

3. Results: 

	city_mpg-price regression: coefficients: [-0.06547185], mean squared error: 0.20, variance score: 0.38

	highway_mpg-price regression: coefficients: [-0.06412873], mean squared error: 0.16, variance score: 0.47

	horsepower-price regression: coefficients: [0.0146964], mean squared error: 0.17, variance score: 0.42

4. We think that the package has found a good solution. From the plots, we can see that the test data points are closely surrounded the linear regression model trained on the training set. It shows that the predictions of the price are generally match the regression results. And the values of MSE and variance are small for each plot. It also proves that the linear regression analysis is good.
