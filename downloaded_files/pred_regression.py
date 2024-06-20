  
def pred_regression(model_path, choose,method, x_test, y_test):
    import joblib
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    import tensorflow as tf
    import numpy as np
    method = method.upper()
    choose = choose.upper()


    if choose == "CELL":
        x_test = np.reshape(x_test, (1,-1))

    if method == "NEURAL":
        model = tf.keras.models.load_model(model_path)

        if choose == "CELL":
            
            pass
        elif choose == "ALL" or choose == "ZERO":
            pass

    else:
        if choose == "CELL":
            model = joblib.load(model_path)
            y_pred = model.predict(x_test)
            return y_pred
        elif choose == "ALL" or choose == "ZERO":
            model = joblib.load(model_path)
            y_pred = model.predict(x_test)
            y_test = y_test.flatten()
            model_rmse = mean_squared_error(y_test, y_pred) ** 0.5
            model_r2 = r2_score(y_test, y_pred)
            return model_rmse, model_r2, y_pred 



