from pyspark.sql import SparkSession
from pyspark.sql.types import (StructField, StructType,
                               IntegerType, StringType)
spark = SparkSession.builder.appName('Basics').getOrCreate()

import pandas as pd
from pyspark.sql.functions import mean, when, col
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline


# --------------------------------------------- Import the Data ---------------------------------------------
average_wages_df = spark.read.csv('DataSet/Average_wages.csv', inferSchema=True, header=True)
housing_prices_df = spark.read.csv('DataSet/Housing_prices.csv', inferSchema=True, header=True)
# df_3 = spark.read.csv('DataSet/portugal_ads_proprieties.csv', inferSchema=True, header=True)
# df_4 = spark.read.csv('DataSet/portugal_houses.csv', inferSchema=True, header=True)

# # --------------------------------------------- Look to the Data ---------------------------------------------
average_wages_df.printSchema()
average_wages_df.show(2)
average_wages_df.describe().show()

housing_prices_df.printSchema()
housing_prices_df.show(2)
housing_prices_df.describe().show()


# ---------------------------- Transform the Data into a final dataframe ---------------------------------------------
# Filter and rename columns in the 'housing_prices' DataFrame
housing_prices = housing_prices_df.filter(housing_prices_df['SUBJECT'] == 'NOMINAL')
housing_prices = housing_prices.select(['LOCATION', 'TIME', 'Value']) \
    .withColumnRenamed('Value', 'Value (Housing - IDX2015)')

# Filter and rename columns in the 'average_wages' DataFrame
average_wages = average_wages_df.select(['LOCATION', 'TIME', 'Value']) \
    .withColumnRenamed('Value', 'Value (Average - USD)')

# Join the two DataFrames on 'LOCATION' and 'TIME'
merged_data = average_wages.join(housing_prices, on=['LOCATION', 'TIME'], how='inner')


# --------------------------------------------- Deal with missing values ---------------------------------------------
# Check for Missing Values in "merged_data" dataframe:
merged_data.filter(merged_data['Value (Average - USD)'].isNull() | merged_data['Value (Housing - IDX2015)'].isNull()).show()

# Calculate the mean for each column with missing values
avg_values = merged_data.agg(*[mean(col(c)).alias(c) for c in merged_data.columns])

# Handle Cases Where There Are No Missing Values: (replace missing values with column average only when there
# are missing values)
merged_data = merged_data.withColumn(
    'Value (Average - USD)',
    when(merged_data['Value (Average - USD)'].isNotNull(), merged_data['Value (Average - USD)'])
    .otherwise(avg_values.first()['Value (Average - USD)'])
)
merged_data = merged_data.withColumn(
    'Value (Housing - IDX2015)',
    when(merged_data['Value (Housing - IDX2015)'].isNotNull(), merged_data['Value (Housing - IDX2015)'])
    .otherwise(avg_values.first()['Value (Housing - IDX2015)'])
)
merged_data.show(5)


# --------------------------------------------- Convert string to numeric ---------------------------------------------
# Convert the strings into integers:
location_indexer = StringIndexer(inputCol='LOCATION',
                                 outputCol='LocIndex')

# Fits or trains the StringIndexer model (location_indexer) on the merged_data DataFrame
location_indexer_model = location_indexer.fit(merged_data)
# Adding a new column (by default named 'LocIndex' based on the outputCol parameter when defining location_indexer).
merged_data = location_indexer_model.transform(merged_data)
merged_data.show(5)


# --------------------------------------------- Scale the Data ---------------------------------------------
# Create a StandardScaler object for each numeric column
numeric_columns = ['LocIndex', 'TIME', 'Value (Average - USD)', 'Value (Housing - IDX2015)']

# Create a VectorAssembler to assemble numeric columns into a single vector column
assembler = VectorAssembler(inputCols=numeric_columns,
                            outputCol='features')

# Create a StandardScaler object
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withMean=True, withStd=True)

# Create a pipeline to first assemble the vectors and then scale them
pipeline = Pipeline(stages=[assembler, scaler])

# Fit the pipeline and transform the data
model = pipeline.fit(merged_data)
scaled_data = model.transform(merged_data)

# Show the scaled data
print("Sample of the 'scaled_features' dataframe: ")
for row in scaled_data.select('scaled_features').limit(5).collect():
    print(row[0])


# ------------------------------------------- Train and Test set creation -------------------------------------------
train_data, test_data = scaled_data.randomSplit([0.7, 0.3])




