from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, when, col, corr
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('Data_pre_processing').getOrCreate()


# --------------------------------------------- Import the Data ---------------------------------------------
average_wages_df = spark.read.csv('DataSet/Average_wages.csv', inferSchema=True, header=True)
housing_prices_df = spark.read.csv('DataSet/Housing_prices.csv', inferSchema=True, header=True)

# # --------------------------------------------- Look to the Data ---------------------------------------------
average_wages_df.printSchema()
average_wages_df.show(2)
average_wages_df.describe().show()

housing_prices_df.printSchema()
housing_prices_df.show(2)
housing_prices_df.describe().show()

# ---------------------------- Transform the Data into the final dataframes ------------------------------------------
# Filter and rename columns in the 'average_wages' DataFrame
average_wages = average_wages_df.select(['LOCATION', 'TIME', 'Value']) \
    .withColumnRenamed('Value', 'Value (Average - USD)')

# Filter and rename columns in the 'housing_prices' DataFrame.
housing_prices = housing_prices_df.filter((housing_prices_df['FREQUENCY'] == 'A'))
housing_prices = housing_prices.select(['LOCATION', 'SUBJECT', col('TIME').cast('int').alias('TIME'), 'Value']) \
    .withColumnRenamed('Value', 'Value (Housing - IDX2015)')

# --------------------------------------------- Deal with missing values ---------------------------------------------
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

# --------------------------------------------- Convert string to numeric ---------------------------------------------
# Convert the strings into integers:
# Fits or trains the StringIndexer model 'location_indexer' for the 'average_wages':
location_indexer_model_hp = StringIndexer(inputCol='LOCATION',
                                          outputCol='LocIndex').fit(housing_prices)
average_wages = location_indexer_model_hp.transform(average_wages)
housing_prices = location_indexer_model_hp.transform(housing_prices)

subject_indexer = StringIndexer(inputCol='SUBJECT',
                                outputCol='SubIndex').fit(housing_prices)
# Fits or trains the StringIndexer model 'location_indexer' for the 'average_wages'
housing_prices = subject_indexer.transform(housing_prices)

average_wages.show(5)
average_wages.printSchema()
housing_prices.show(5)
housing_prices.printSchema()

# --------------------------------------------- Check Correlation ---------------------------------------------
# For both data sets, bellow it's being checked the correlation between value and time/location.
average_wages.select(corr('Value (Average - USD)', 'TIME')).show()
average_wages.select(corr('Value (Average - USD)', 'LocIndex')).show()
# Same here:
housing_prices.select(corr('Value (Housing - IDX2015)', 'TIME')).show()
housing_prices.select(corr('Value (Housing - IDX2015)', 'LocIndex')).show()

# -------------------------------------- Scale the Data for the "features" -------------------------------------------
# Create a StandardScaler object for each numeric column
numeric_columns_aw = ['LocIndex', 'TIME']
numeric_columns_hp = ['LocIndex', 'SubIndex', 'TIME']

# Create a VectorAssembler to assemble numeric columns into a single vector column
features_assembler_aw = VectorAssembler(inputCols=numeric_columns_aw, outputCol='features_aw')
features_assembler_hp = VectorAssembler(inputCols=numeric_columns_hp, outputCol='features_hp')
label_assembler_aw = VectorAssembler(inputCols=["Value (Average - USD)"], outputCol="label_aw")
label_assembler_hp = VectorAssembler(inputCols=["Value (Housing - IDX2015)"], outputCol="label_hp")

# Create a StandardScaler object
features_scaler_aw = StandardScaler(inputCol='features_aw',
                                    outputCol='scaled_features_aw', withMean=True, withStd=True)
features_scaler_hp = StandardScaler(inputCol='features_hp',
                                    outputCol='scaled_features_hp', withMean=True, withStd=True)
label_scaler_aw = StandardScaler(inputCol='label_aw',
                                 outputCol='scaled_label_aw', withMean=True, withStd=True)
label_scaler_hp = StandardScaler(inputCol='label_hp',
                                 outputCol='scaled_label_hp', withMean=True, withStd=True)

# Create a pipeline to first assemble the vectors and then scale them
features_pipeline_aw = Pipeline(stages=[features_assembler_aw, features_scaler_aw])
label_pipeline_aw = Pipeline(stages=[label_assembler_aw, label_scaler_aw])
features_pipeline_hp = Pipeline(stages=[features_assembler_hp, features_scaler_hp])
labels_pipeline_hp = Pipeline(stages=[label_assembler_hp, label_scaler_hp])

# Fit the pipeline and transform the features on average_wages:
feature_scaler_model_aw = features_pipeline_aw.fit(average_wages)
pre_scaled_data_aw = feature_scaler_model_aw.transform(average_wages)
# Fit the pipeline and transform the labels on average_wages:
label_scaler_model_aw = label_pipeline_aw.fit(pre_scaled_data_aw)
scaled_data_aw = label_scaler_model_aw.transform(pre_scaled_data_aw)

# Fit the pipeline and transform the features on housing_prices:
model_hp = features_pipeline_hp.fit(housing_prices)
pre_scaled_data_hp = model_hp.transform(housing_prices)
# Fit the pipeline and transform the labels on housing_prices:
label_scaler_model_hp = labels_pipeline_hp.fit(pre_scaled_data_hp)
scaled_data_hp = label_scaler_model_hp.transform(pre_scaled_data_hp)

# Show the scaled data
print("Sample of the 'scaled_features_aw' dataframe: ")
scaled_data_aw.show(truncate=False)
print("Sample of the 'scaled_features_hp' dataframe: ")
scaled_data_hp.show(truncate=False)

# ------------------------------------------- Train and Test set creation -------------------------------------------
ml_model_data_aw = scaled_data_aw.select("scaled_features_aw", "scaled_label_aw")
ml_model_data_hp = scaled_data_hp.select("scaled_features_hp", "scaled_label_hp")

ml_model_data_aw.show(10, truncate=False)
ml_model_data_hp.show(10, truncate=False)

train_data_aw, test_data_aw = scaled_data_aw.randomSplit([0.7, 0.3])
train_data_hp, test_data_hp = scaled_data_hp.randomSplit([0.7, 0.3])


# ------------------------------------------- Call the ML_model -------------------------------------------

# AW Model set:
aw_lr = LinearRegression(featuresCol='scaled_features_aw',
                         labelCol='scaled_label_aw',
                         predictionCol='prediction_aw',
                         maxIter=1000)
aw_lr_model = aw_lr.fit(train_data_aw)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(aw_lr_model.coefficients,
                                              aw_lr_model.intercept))


# HP Model set:
hp_lr = LinearRegression(featuresCol='scaled_features_hp',
                         labelCol='scaled_label_hp',
                         predictionCol='prediction_hp',
                         maxIter=1000)
hp_lr_model = hp_lr.fit(train_data_hp)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(hp_lr_model.coefficients,
                                              hp_lr_model.intercept))







# --------------------
from pyspark.sql.functions import lit

# Create a DataFrame with the 'Time' values you want to predict for (2023 to 2027)
prediction_years_df = spark.createDataFrame([(2023,), (2024,), (2025,), (2026,), (2027,)], ["TIME"])

# Add a new column with 'LocIndex' for 'AUS' (assuming 'AUS' is the location you want to predict for)
prediction_years_df = prediction_years_df.withColumn("LOCATION", lit("AUS"))
prediction_years_df = location_indexer_model_hp.transform(prediction_years_df)
prediction_years_df.show()

# Assemble the features for prediction
prediction_years_df = feature_scaler_model_aw.transform(prediction_years_df)

# Use the trained model to make predictions
predictions = aw_lr_model.transform(prediction_years_df)

# Show the predictions
predictions.select("LOCATION", "TIME", "prediction_aw").show()