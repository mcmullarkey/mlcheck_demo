
import pandas
import sklearn
import sklearn.model_selection
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def function_start():
    print('---')

def function_end():
    print('---')

############################################
function_start()
RANDOM_STATE = 10

df = pandas.read_csv('data/weather/w_IntegerX100Temp_2col_95000.csv')

# Split data
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

# Train
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
                                    X, Y, test_size=0.2, random_state=RANDOM_STATE)
regression = linear_model.LinearRegression()
regression.fit(X_train, Y_train)

# Evaluate
pred = regression.predict(X_test)
print('Test Set RMSE: ' + str(mean_squared_error(Y_test, pred)))

function_end()
############################################
