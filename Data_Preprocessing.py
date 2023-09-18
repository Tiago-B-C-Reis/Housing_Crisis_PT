from pyspark.sql import SparkSession
from pyspark.sql.types import (StructField, StructType,
                               IntegerType, StringType)
spark = SparkSession.builder.appName('Basics').getOrCreate()

df_1 = spark.read.csv('DataSet/Average_wages.csv', inferSchema=True, header=True)
df_2 = spark.read.csv('DataSet/Housing_prices.csv', inferSchema=True, header=True)
df_3 = spark.read.csv('DataSet/portugal_ads_proprieties.csv', inferSchema=True, header=True)
df_4 = spark.read.csv('DataSet/portugal_houses.csv', inferSchema=True, header=True)

df_1.printSchema()
df_1.describe().show()

df_2.printSchema()
df_2.describe().show()

df_3.printSchema()
df_3.describe().show()

df_4.printSchema()
df_4.describe().show()

# df_1.show()
# print(df.head(3)[0])




