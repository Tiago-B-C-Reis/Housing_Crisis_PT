from pyspark.sql import SparkSession
from pyspark.sql.types import (StructField, StructType,
                               IntegerType, StringType)
spark = SparkSession.builder.appName('scratch').getOrCreate()

import pandas as pd
from pyspark.sql.functions import mean, when, col, corr
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from ML_model import ml_model


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
housing_prices.describe().show()

# --------------------------------------------- Check Correlation ---------------------------------------------
# This 2 features are 91,52% correlated, which means that one strongly
# affects the other.
average_wages.select(corr('Value (Average - USD)', 'TIME')).show()
average_wages.select(corr('Value (Average - USD)', 'LOCATION')).show()
# Same here:
housing_prices.select(corr('Value (Housing - IDX2015)', 'TIME')).show()
housing_prices.select(corr('Value (Housing - IDX2015)', 'LOCATION')).show()

# --------------------------------------------- Scale the Data ---------------------------------------------
# Create a StandardScaler object for each numeric column
numeric_columns_aw = ['LocIndex', 'TIME', 'Value (Average - USD)']
numeric_columns_hp = ['LocIndex', 'SubIndex', 'TIME', 'Value (Housing - IDX2015)']

# Create a VectorAssembler to assemble numeric columns into a single vector column
assembler_aw = VectorAssembler(inputCols=numeric_columns_aw, outputCol='features_aw')
assembler_hp = VectorAssembler(inputCols=numeric_columns_hp, outputCol='features_hp')

# Create a StandardScaler object
scaler_aw = StandardScaler(inputCol='features_aw', outputCol='scaled_features_aw', withMean=True, withStd=True)
scaler_hp = StandardScaler(inputCol='features_hp', outputCol='scaled_features_hp', withMean=True, withStd=True)

# Create a pipeline to first assemble the vectors and then scale them
pipeline_aw = Pipeline(stages=[assembler_aw, scaler_aw])
pipeline_hp = Pipeline(stages=[assembler_hp, scaler_hp])

# Fit the pipeline and transform the data
model_aw = pipeline_aw.fit(average_wages)
scaled_data_aw = model_aw.transform(average_wages)

model_hp = pipeline_hp.fit(housing_prices)
scaled_data_hp = model_hp.transform(housing_prices)

# Show the scaled data
print("Sample of the 'scaled_features_aw' dataframe: ")
scaled_data_aw.select('scaled_features_aw').limit(5).show(truncate=False)

print("Sample of the 'scaled_features_hp' dataframe: ")
scaled_data_hp.select('scaled_features_hp').limit(5).show(truncate=False)

# ------------------------------------------- Train and Test set creation -------------------------------------------
train_data, test_data = scaled_data_hp.randomSplit([0.7, 0.3])

# ------------------------------------------- Call the ML_model -------------------------------------------
ml_model(train_data, test_data)
