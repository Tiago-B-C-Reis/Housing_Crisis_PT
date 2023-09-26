from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('ML_model').getOrCreate()

def ml_model_aw(train_data, test_data):


    # Model set:
    lr = LinearRegression(featuresCol='features_aw',
                          labelCol='crew',
                          predictionCol='prediction',
                          maxIter=1000)
    lr_model = lr.fit(train_data)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: {} Intercept: {}".format(lr_model.coefficients,
                                                  lr_model.intercept))


    return ml_model_aw_results


def ml_model_hp(train_data, test_data):


    # Model set:
    lr = LinearRegression(featuresCol='features',
                          labelCol='crew',
                          predictionCol='prediction',
                          maxIter=1000)
    lr_model = lr.fit(train_data)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: {} Intercept: {}".format(lr_model.coefficients,
                                                  lr_model.intercept))


    return ml_model_hp_results