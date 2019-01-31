"""
===============================================================================
----------------------------   Spark-ML HOME WORK   ---------------------------
===============================================================================

When creating ML models, the conceot of efficiency has two sides:
    (1) the time dedicated by the analyst to build the model
    (2) the computer tima and resources needed by the final model

Efficiency is a combination of both

In this project, you are asked to optimize the second. Spark is the best tool
to build models over massive datasets

If you need to create Spark+Python Machine Learning models that "run fast" on the
cluster, you must avoid using Python code or working with RRD+python. Try to use
the already existing methods that do what you need (do not reinvent the wheel)

Therefore try to use the implemented object+methods inside the Spark SQL and ML
modules. They are very fast, because it is complied Java/Scala code. Try to
use: DataFrames, Feature Transfomers, Estimators, Pipelines, GridSearch, CV, ...

For this homework, you are asked to create a classification model that:
    (1) uses ALL variables in the dataset (homework TRAIN.xlsx) to predict label "loan_status"
    (2) your solution must be a python scripts that:
        (3.1) reads the "bank.csv" file, transform and select variables as you wish
            ( but start using use ALL)
        (3.2) tune the model and hyperparameters using gridsearch (using a minimun
                of 10 sets of hyperparameters ans using 3 folds for validation)
        (3.3) compute the AUC for the test set
    (3) Your work will be evaluated under the following scoring schema
        (30%) a clean, clever and efficient use of the Python and Spark objects
                and methods
        (30%) timing that takes your code to run on my computer (I will time it)
        (40%) AUC on the test set (I will use my own train and test sets)

Enjoy it and best of luck!!

"""

import os
import sys

os.environ['SPARK_HOME'] = 'C:\spark-2.1.1-bin-hadoop2.7'

# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

#Add the following paths to the system path. Please check your installation
#to make sure that these zip files actually exist. The names might change
#as versions change.
sys.path.insert(0,os.path.join(SPARK_HOME,"python"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","pyspark.zip"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","py4j-0.10.4-src.zip")) #es el interprete entre java y python

#Initialize SparkSession and SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext



#Create a Spark Session
MySparkSession = SparkSession.builder.config("spark.executor.memory", "6g").config("spark.cores.max","4").getOrCreate()
    

#Get the Spark Context from Spark Session    
MySparkContext = MySparkSession.sparkContext



##CARGAMOS LOS DATOS Y DEFINIMOS EL SCHEMA


import pyspark.sql.types as typ
values = [
    ('ID', typ.IntegerType()),
    ('loan_amnt', typ.IntegerType()),
    ('term',typ.StringType() ),
    ('int_rate', typ.FloatType()),
    ('installment', typ.FloatType()),
    ('emp_length', typ.StringType()),
    ('home_ownership', typ.StringType()),
    ('annual_inc', typ.FloatType()),
    ('purpose', typ.StringType()),
    ('title', typ.StringType()),
    ('STATE', typ.StringType()),
    ('delinq_2yrs', typ.IntegerType()),
    ('revol_bal', typ.IntegerType()),
    ('revol_util',  typ.FloatType()),
    ('total_pymnt', typ.FloatType()),
    ('loan_status', typ.StringType())
]


schema = typ.StructType( [typ.StructField(e[0], e[1], True) for e in values] ) 

datos = MySparkSession.read.csv("C:/Users/rsalas/Documents/Master BIGDATA/Módulo 2- Arquitecturas de información\Machine Learning para Spark/homework/homework TRAIN.csv",schema=schema, header=True,sep=";")

datos.take(5)
datos.first()

#¿Cuántos datos tengo?
datos.count()

#Columnas
datos.columns

#Previo análisis descriptivo
datos.describe().show()



##MISSING VALUES
from pyspark.sql.functions import isnan, when, count, col

#NaNs
datos.select([count(when(isnan(c), c)).alias(c) for c in datos.columns]).show()

#Observamos que no hay Nas.


#Nulos
datos.select([count(when(col(c).isNull(), c)).alias(c) for c in datos.columns]).show()

#Hay valores nulos, ¿qué porcentaje de unknown hay en cada columna?

import pyspark.sql.functions as fn

datos.agg(*[ (1 - (fn.count(c) / fn.count('*'))).alias(c + '_missing')
    for c in datos.columns ]).show()

#Eliminamos todas las filas con missing, pues únicamente suponen el 4% del dataset

datossinna = datos.na.drop() # eliminamos los NA


##FEATURE ENGINEER
#Eliminamos las variables 'ID' y 'title'. Además vamos a crearnos una nueva variable que va a corresponder al cociente entre 'total_pymnt' y 'loan_amnt'

datos2 =datossinna.drop('ID','title').withColumn('int_rate', fn.col('int_rate')*0.01).withColumn("coctotal", fn.col('total_pymnt')/fn.col('loan_amnt')).drop('total_payment').drop('loan_amnt')




##EXPLORATORY DATA ANALYSIS
cat_vars = ['term', 'emp_length', 'home_ownership', 'purpose','STATE']
set(cat_vars)

# Principales estadísticos para variables numéricas
datos2.describe( [c for c in datos2.columns if c not in cat_vars] ).show()





##PREPARE DATA FOR ML

#Observamos el número de niveles o valores distintos que presentan las variables
#for col in datos2.columns:
 #   print(col, datos2.select(col).distinct().count())


#La variable STATE nos va a dar problemas ya que tiene muchos niveles y algunos de ellos tienen pocas observaciones. 
#Agrupamos estos en un nivel que agregue los que menos observaciones tienen.
        
datos2.toPandas()['home_ownership'].value_counts()

datos2.toPandas()['STATE'].value_counts()

catSTATE = ['AK','SD','VT','TN','MS','IN','IA','NE','ID','ME','DE','MT','WY']
cathome_ownership = ['OTHER', 'NONE']


from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


state_udf = udf(lambda STATE: "PEQUE" if STATE in catSTATE else STATE, StringType())
home_ownership_udf = udf(lambda home_ownership: 'OTHER' if home_ownership in cathome_ownership else home_ownership, StringType())
datos2=datos2.withColumn("STATE", state_udf(datos2.STATE)).withColumn('home_ownership',home_ownership_udf(datos2.home_ownership))



#Cargamos las librerías necesarias

from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
import pyspark.ml.feature as ft
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier


#Binarizamos las variables categóricas, tienen más de dos factores

#Transformers
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in cat_vars]

#target
indexers += [StringIndexer(inputCol='loan_status', outputCol='loan_status_Index')]

encoders = [OneHotEncoder(dropLast=False, inputCol=column+"_index", outputCol=column+"_onehot") for column in cat_vars]

#Definimos las variables del modelo
features =['int_rate','installment','annual_inc','revol_bal','revol_util','coctotal','delinq_2yrs','term_onehot','emp_length_onehot','purpose_onehot','STATE_onehot','home_ownership_onehot']

vectorassembler_stage = VectorAssembler(inputCols=features, outputCol='features')



#Creamos el pipeline
all_stages = indexers + encoders + [vectorassembler_stage]
pipeline = Pipeline(stages=all_stages)


# Fit the pipeline to dataset
datos_df = pipeline.fit(datos2).transform(datos2)


#Dividimos nuestros datos en train y test
training, test = datos_df.randomSplit([0.8, 0.2], seed=1234)



### GB
#El modelo elegido va ser Gradient Boosting, con el que mejor AUC hemos obtenido en un menor tiempo

from time import time

t0 = time()


#Estimator
gbt = GBTClassifier(labelCol="loan_status_Index", featuresCol="features")


evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", \
                    labelCol="loan_status_Index",metricName="areaUnderROC")

#Usamos paramgridbuilder para construir un grid de hiperparámetros. Tenemos 3x3x2=18 modelos
#para entrenar y validar

paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [2, 3, 4,5]) \
    .addGrid(gbt.maxIter, [2,3,4]) \
    .build()

#Cross validation con 3 folds
crossval_gbt = CrossValidator(estimator = gbt,
                          estimatorParamMaps = paramGrid_gbt,
                          evaluator = evaluator,
                          numFolds = 3) 

# Run cross-validation, and returns the best model.
cvModel_gbt= crossval_gbt.fit(training)

# Predecimos en test
predictionCV_gbt = cvModel_gbt.transform(test)
print(evaluator.evaluate(predictionCV_gbt))

tt = time() - t0
print("Classifier trained in {} seconds".format(round(tt,3)))




results = [([{key.name: paramValue} for key, paramValue in zip(params.keys(), params.values())], metric) for params, metric in zip(cvModel_gbt.getEstimatorParamMaps(), cvModel_gbt.avgMetrics)] 
results


#0.9168430646318525
#Classifier trained in 433.561 seconds
#0.9018088175695451
#Classifier trained in 230.147 seconds
