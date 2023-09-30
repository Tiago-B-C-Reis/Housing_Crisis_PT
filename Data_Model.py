from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, when, col, corr
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('Data_pre_processing').getOrCreate()


# --------------------------------------------- Import the Data -----------------------------------------------------
average_wages_df = spark.read.csv('DataSet/Average_wages.csv', inferSchema=True, header=True)
housing_prices_df = spark.read.csv('DataSet/Housing_prices.csv', inferSchema=True, header=True)

# # --------------------------------------------- Look to the Data --------------------------------------------------
average_wages_df.printSchema()
average_wages_df.show(2)
average_wages_df.describe().show()

housing_prices_df.printSchema()
housing_prices_df.show(2)
housing_prices_df.describe().show()

# ---------------------------- Transform the Data into the final dataframes -----------------------------------------
# Filter and rename columns in the 'average_wages' DataFrame
average_wages = average_wages_df.select(['LOCATION', 'TIME', 'Value']) \
    .withColumnRenamed('Value', 'Value (Average - USD)')

# Filter and rename columns in the 'housing_prices' DataFrame.
housing_prices = housing_prices_df.filter((housing_prices_df['FREQUENCY'] == 'A'))
housing_prices = housing_prices.select(['LOCATION', 'SUBJECT', col('TIME').cast('int').alias('TIME'), 'Value']) \
    .withColumnRenamed('Value', 'Value (Housing - IDX2015)')

# --------------------------------------------- Deal with missing values --------------------------------------------
# Check for Missing Values in "merged_data" dataframe:
average_wages.filter(average_wages['Value (Average - USD)'].isNull()).show()
housing_prices.filter(housing_prices['Value (Housing - IDX2015)'].isNull()).show()

# Calculate the mean for each column with missing values
avg_values_aw = average_wages.agg(*[mean(col(c)).alias(c) for c in average_wages.columns])
avg_values_hp = housing_prices.agg(*[mean(col(c)).alias(c) for c in housing_prices.columns])

# Handle Cases Where There Are No Missing Values: (replace missing values with column average only when there
# are missing values)
average_wages = average_wages.withColumn(
    'Value (Average - USD)',
    when(average_wages['Value (Average - USD)'].isNotNull(), average_wages['Value (Average - USD)'])
    .otherwise(avg_values_aw.first()['Value (Average - USD)'])
)
housing_prices = housing_prices.withColumn(
    'Value (Housing - IDX2015)',
    when(housing_prices['Value (Housing - IDX2015)'].isNotNull(), housing_prices['Value (Housing - IDX2015)'])
    .otherwise(avg_values_hp.first()['Value (Housing - IDX2015)'])
)

# -------------------------------------------- Convert string to numeric --------------------------------------------
# Convert the strings into integers:
# Fits or trains the StringIndexer model 'location_indexer' for the 'average_wages':
location_indexer_model_hp = StringIndexer(inputCol='LOCATION',
                                          outputCol='LocIndex').fit(housing_prices)
average_wages = location_indexer_model_hp.transform(average_wages)
housing_prices = location_indexer_model_hp.transform(housing_prices)

subject_indexer_model_hp = StringIndexer(inputCol='SUBJECT',
                                outputCol='SubIndex').fit(housing_prices)
# Fits or trains the StringIndexer model 'location_indexer' for the 'average_wages'
housing_prices = subject_indexer_model_hp.transform(housing_prices)

average_wages.show(5)
average_wages.printSchema()
housing_prices.show(5)
housing_prices.printSchema()

# --------------------------------------------- Check Correlation ---------------------------------------------------
# For both data sets, bellow it's being checked the correlation between value and time/location.
average_wages.select(corr('Value (Average - USD)', 'TIME')).show()
average_wages.select(corr('Value (Average - USD)', 'LocIndex')).show()
# Same here:
housing_prices.select(corr('Value (Housing - IDX2015)', 'TIME')).show()
housing_prices.select(corr('Value (Housing - IDX2015)', 'LocIndex')).show()

# -------------------------------------- Scale the Data for the "features" ------------------------------------------
# Create a StandardScaler object for each numeric column
numeric_columns_aw = ['LocIndex', 'TIME']
numeric_columns_hp = ['LocIndex', 'SubIndex', 'TIME']

# Create a VectorAssembler to assemble numeric columns into a single vector column
features_assembler_aw = VectorAssembler(inputCols=numeric_columns_aw, outputCol='features_aw')
features_assembler_hp = VectorAssembler(inputCols=numeric_columns_hp, outputCol='features_hp')

# Create a StandardScaler object
features_scaler_aw = StandardScaler(inputCol='features_aw',
                                    outputCol='scaled_features_aw', withMean=True, withStd=True)
features_scaler_hp = StandardScaler(inputCol='features_hp',
                                    outputCol='scaled_features_hp', withMean=True, withStd=True)

# Create a pipeline to first assemble the vectors and then scale them
features_pipeline_aw = Pipeline(stages=[features_assembler_aw, features_scaler_aw])
features_pipeline_hp = Pipeline(stages=[features_assembler_hp, features_scaler_hp])

# Fit the pipeline and transform the features on average_wages:
feature_scaler_model_aw = features_pipeline_aw.fit(average_wages)
scaled_data_aw = feature_scaler_model_aw.transform(average_wages)

# Fit the pipeline and transform the features on housing_prices:
feature_scaler_model_hp = features_pipeline_hp.fit(housing_prices)
scaled_data_hp = feature_scaler_model_hp.transform(housing_prices)

# Show the scaled data
print("Sample of the 'scaled_data_aw' dataframe: ")
scaled_data_aw.show(truncate=False)
print("Sample of the 'scaled_data_hp' dataframe: ")
scaled_data_hp.show(truncate=False)

# ------------------------------------------- Train and Test set creation -------------------------------------------
ml_model_data_aw = scaled_data_aw.select("scaled_features_aw", "Value (Average - USD)")
ml_model_data_hp = scaled_data_hp.select("scaled_features_hp", "Value (Housing - IDX2015)")

ml_model_data_aw.show(10, truncate=False)
ml_model_data_hp.show(10, truncate=False)

train_data_aw, test_data_aw = scaled_data_aw.randomSplit([0.7, 0.3])
train_data_hp, test_data_hp = scaled_data_hp.randomSplit([0.7, 0.3])

# ----------------------------------------- ML_models (Linear regression) -------------------------------------------
# AW Model set:
aw_lr = LinearRegression(featuresCol='scaled_features_aw',
                         labelCol='Value (Average - USD)',
                         predictionCol='prediction_aw',
                         maxIter=1000)
aw_lr_model = aw_lr.fit(train_data_aw)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(aw_lr_model.coefficients,
                                              aw_lr_model.intercept))


# HP Model set:
hp_lr = LinearRegression(featuresCol='scaled_features_hp',
                         labelCol='Value (Housing - IDX2015)',
                         predictionCol='prediction_hp',
                         maxIter=1000)
hp_lr_model = hp_lr.fit(train_data_hp)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(hp_lr_model.coefficients,
                                              hp_lr_model.intercept))


# ----------------------------------------- ML model evaluation -----------------------------------------------------

# Evaluate the model on the test set (for the 'aw_lr_model'):
test_results_aw = aw_lr_model.evaluate(test_data_aw)
test_results_aw.residuals.show()
# Evaluate the model on the test set (for the 'hp_lr_model'):
test_results_hp = hp_lr_model.evaluate(test_data_hp)
test_results_hp.residuals.show()


# This is the 'Root Mean Square Error'
print("RMSE: (aw) {}".format(test_results_aw.rootMeanSquaredError))
print("RMSE: (hp) {}\n".format(test_results_aw.rootMeanSquaredError))

#
print("MSE: (aw) {}".format(test_results_aw.meanSquaredError))
print("MSE: (hp) {}\n".format(test_results_hp.meanSquaredError))

#
print("R2: (aw) {}".format(test_results_aw.r2))
print("R2: (hp) {}".format(test_results_hp.r2))


# ----------------------------------------- ML model predictions -----------------------------------------------------

# Get predictions for the test data using the "Value (Average - USD)" model
predictions_aw = aw_lr_model.transform(scaled_data_aw)
# Get predictions for the test data using the "Value (Housing - IDX2015)" model
predictions_hp = hp_lr_model.transform(scaled_data_hp)

# Show the predictions
predictions_aw.show(5)
predictions_hp.show(5)

# --------------------------------------- ML model 5 years predictions -----------------------------------------------
# Define the range of years you want to add (from 2023 to 2030)
years_to_add = list(range(2023, 2031))

# Get unique locations from the original dataset
unique_locations = scaled_data_aw.select("LOCATION").distinct()
unique_subjects = scaled_data_hp.select("SUBJECT").distinct()

# Create a DataFrame with all combinations of locations and years (average_wages)
predictions_data_aw = unique_locations.crossJoin(
    spark.createDataFrame([(year,) for year in years_to_add], ["TIME"])
)
# Create a DataFrame with all combinations of locations and years (housing_prices)
predictions_data_hp = unique_locations.crossJoin(unique_subjects).crossJoin(
    spark.createDataFrame([(year,) for year in years_to_add], ["TIME"])
)

#
predictions_data_aw = location_indexer_model_hp.transform(predictions_data_aw)
#
predictions_data_hp = location_indexer_model_hp.transform(predictions_data_hp)
predictions_data_hp = subject_indexer_model_hp.transform(predictions_data_hp)

# Assemble the features for prediction
predictions_data_aw = feature_scaler_model_aw.transform(predictions_data_aw)
predictions_data_hp = feature_scaler_model_hp.transform(predictions_data_hp)

# Use the trained model to make predictions
predictions_aw = aw_lr_model.transform(predictions_data_aw)
predictions_hp = hp_lr_model.transform(predictions_data_hp)

# Show the expanded DataFrame
predictions_aw.orderBy("LOCATION", "TIME").show()
predictions_hp.orderBy("LOCATION", "SUBJECT", "TIME").show()


# ---------------------------- Add the 5 years of predictions to the final datasets ----------------------------------


# Filter rows from the second table where TIME is between 2023 and 2030
filtered_second_table = predictions_hp.filter((predictions_hp['TIME'] >= 2023) & (predictions_hp['TIME'] <= 2030))

# Select the columns you want to keep in both DataFrames
# Assuming you want to keep LOCATION, SUBJECT, TIME, Value (Housing - IDX2015)
first_table = scaled_data_hp.select('LOCATION', 'SUBJECT', 'TIME', 'Value (Housing - IDX2015)')
filtered_second_table = filtered_second_table.select('LOCATION', 'SUBJECT', 'TIME', 'prediction_hp')

# Union the two DataFrames to combine them
final_table = first_table.union(filtered_second_table)

# Show the final combined DataFrame
final_table.show(60)
