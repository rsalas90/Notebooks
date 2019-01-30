
# coding: utf-8

# # SOLUCIÓN ESTÁNDAR EXTRACCIÓN DE FEATURES CON FEATURETOOLS (PYSPARK)

# ### Objetivo

# El objetivo de este notebook es proporcionar un método estándar para la extracción de nuevas variables utilizando la librería **Featuretools** en PySpark.

# Comenzamos inicializando nuestra SparkSession y Spark Context.

# In[578]:


from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.context import SQLContext
sqlContext = SQLContext(sc)

spark = SparkSession.builder.master("local[2]").appName("MiPrimer").config("spark.executor.memory", "6g").config("spark.cores.max","4").getOrCreate()
sc = spark.sparkContext


# Importamos las librerías que vamos a necesitar:

# In[571]:


import pandas as pd
import numpy as np

import featuretools as ft
import featuretools.variable_types as vtypes

import datetime 
import pyspark.sql.functions as F
from pyspark.sql.functions import unix_timestamp, from_unixtime, to_date
from pyspark.sql.functions import pandas_udf, PandasUDFType, udf


# ### Ejemplo 1:

# Cargamos los datos que están disponibles en la librería **Featuretools**

# In[572]:


data = ft.demo.load_mock_customer()

#Cargamos un único dataset
customer_df = data['customers'] 

#Creamos el EntitySet
es = ft.EntitySet('name', {'customers': (customer_df, 'customer_id')})

    #Es lo mismo:
    # es = ft.EntitySet(id="name")
    # es = es.entity_from_dataframe(entity_id="customers", index='customer_id', dataframe=customer_df)

#Generamos el dataframe con las nuevas variables:
feature_matrix, feats = ft.dfs(entityset=es, target_entity='customers', max_depth=2)


# A continuación creamos un DataFrame de spark a partir de nuestro dataframe:

# In[573]:


customer_sp = spark.createDataFrame(customer_df)
target_schema = spark.createDataFrame(feature_matrix.reset_index()).schema


# In[574]:


target_schema


# Este es el dataframe que hemos creado:

# In[575]:


customer_sp.show()


# Para poder realizar la extracción de variables necesitamos la función @pandas_udf de pyspark.

# La función *@pandas_udf* lleva 3 atributos:  
# - Function: función definida por el usuario  
# - ReturnType: el tipo de salida que devuelve
# - FunctionType: puede ser **Scalar**, si define una transformación de una o más pandas series a un panda series o **Grouped Map**, si define una transformación de un pandas DataFrame a un pandas Dataframe. En este caso debera incluirse el *schema* en el ReturnType.

# Pasos:
# 1. Cogemos una muestra (unas 5 filas) del dataframe de spark si es muy grande y convertimos la muestra a pandas (toPandas()).  
# 2. Vamos a realizar en python la parte de featuretools para obtener las nuevas features creadas y generar el schema que pasaremos después a la función pandas_udf.  
# 3. Para asegurarnos que se generan las mismas variables, generamos/guardarmos las features que devuelve la feature_matrix y usamos la función *calculate_feature_matrix* de **Featuretools**.

# In[576]:


@pandas_udf(target_schema, PandasUDFType.GROUPED_MAP)   
def generate_features(df_sp):
    es = ft.EntitySet('name', {'customers': (df_sp, 'customer_id')})
    return ft.calculate_feature_matrix(feats, es).reset_index()


# In[577]:


customer_ft = customer_sp.groupby("customer_id").apply(generate_features)
customer_ft.toPandas()


# Observamos como hemos creado nuevas variables en nuestro dataframe de pyspark.

# Observación: realizar el adecuado preprocesamiento de datos previo a ejecutar featuretools: limpieza de nulls, conversión de tipos...

# ### Ejemplo 2

# En el siguiente ejemplo tenemos un dataset de venta de coches el cual vamos a leer con spark:

# In[637]:


Ventas_coches = spark.read.csv("Car_sales.csv", header=True)


# In[638]:


Ventas_coches.printSchema()


# Observamos como el esquema de las variables no corresponde con el tipo que es cada una. Procedemos a cambiar el tipo de cada una de ellas, fijándonos bien en la columna *Latest_Launch* que es de tipo fecha.

# In[639]:


Ventas_coches = Ventas_coches.withColumn('Sales_in_thousands', F.col('Sales_in_thousands').cast('float')).withColumn('__year_resale_value', F.col('__year_resale_value').cast('float')).withColumn('Price_in_thousands', F.col('Price_in_thousands').cast('float')).withColumn('Engine_size', F.col('Engine_size').cast('float')).withColumn('Horsepower', F.col('Horsepower').cast('int')).withColumn('Wheelbase', F.col('Wheelbase').cast('float')).withColumn('Width', F.col('Width').cast('float')).withColumn('Length', F.col('Length').cast('float')).withColumn('Curb_weight', F.col('Curb_weight').cast('float')).withColumn('Fuel_capacity', F.col('Fuel_capacity').cast('float')).withColumn('Fuel_efficiency', F.col('Fuel_efficiency').cast('float')).withColumn('Latest_Launch',to_date(unix_timestamp('Latest_Launch', "d/M/yyyy").cast('timestamp'))).withColumn('Power_perf_factor', F.col('Power_perf_factor').cast('float'))


# Una vez realizado, vemos como todas las variables son del tipo correspondiente.

# In[640]:


Ventas_coches.printSchema()


# Para ejecutar *featuretools* debe haber un Id único para cada registro. Creamos una columna adicional que nos genere este Id.

# In[641]:


from pyspark.sql.functions import monotonically_increasing_id
Ventas_coches = Ventas_coches.withColumn('Id', monotonically_increasing_id())


# Como hemos comentado, el dataframe debe estar limpio de registros nulos. Veamos cuántos tenemos:

# In[642]:


Ventas_coches.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in Ventas_coches.columns]).show()
# Ventas_coches.select([F.count(F.when(F.isnan(c), c)).alias(c) for c in Ventas_coches.columns]).show()


# Para este caso concreto, vamos a filtrar por aquellos registros que no son nulos y elimninar las columnas que no nos son necesarias para generar nuevas variables.

# In[643]:


Ventas_coches2 = Ventas_coches.where(Ventas_coches.Latest_Launch.isNotNull()).drop('__year_resale_value').drop('Engine_size').drop('Fuel_efficiency').drop('Price_in_thousands').drop('Power_perf_factor')


# In[644]:


Ventas_coches2.count()


# In[645]:


Ventas_coches2.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in Ventas_coches2.columns]).show()


# Generamos la muestra convirtiendo a pandas nuestro dataframe.

# In[646]:


# Ventas_coches_prueba = Ventas_coches.select(['Id', 'Manufacturer', 'Latest_Launch']).toPandas().head(5)
Ventas_coches_prueba = Ventas_coches2.toPandas().head(5)


# In[690]:


Ventas_coches_prueba


#   
#   Creamos el EntitySet con un id cualquiera, en este caso 'name'. Añadimos la nueva entidad que corresponde con el dataframe de muestra que hemos generado.  
# Como index le pasamos la columna *Id* previamente creada y como entity_id el nombre que queramos para esa entidad.

# In[648]:


es = ft.EntitySet(id="name")
es = es.entity_from_dataframe(entity_id="dataprueba", index='Id', dataframe=Ventas_coches_prueba)

#Generamos el dataframe con las nuevas variables:
feature_matrix, feats = ft.dfs(entityset=es, target_entity='dataprueba', max_depth=2)


# In[649]:


feature_matrix


# Una vez hemos creado la matriz de nuevas variables la convertimos a un DataFrame de spark y nos guardamos el schema que necesitaremos después para pasarlo como atributo en la función @pandas_udf.

# In[650]:


# customer_sp = spark.createDataFrame(customer_df)
target_schema = spark.createDataFrame(feature_matrix.reset_index()).schema


# In[651]:


target_schema


# Definimos la función pasandole los argumentos:

# In[652]:


@pandas_udf(target_schema, PandasUDFType.GROUPED_MAP)   
def generate_features(df_sp):
    es = ft.EntitySet('name', {'dataprueba': (df_sp, 'Id')})
    return ft.calculate_feature_matrix(feats, es).reset_index()


# Aplicamos la función sobre la columna *Id* del dataframe de spark:

# In[653]:


data_ft = Ventas_coches2.groupby("Id").apply(generate_features)


# In[654]:


data_ft.show(5)


# Observamos que hemos obtenido las nuevas variables correspondientes a la fecha que era la única que podía generar nuevas features.

# ### Próximos pasos:

# Ahora nos preguntamos cómo agrupar nuestro dataframe de manera que las features generadas sean agregaciones y no transformaciones.  
# Realizamos los mismos pasos que antes:

# In[720]:


Ventas_coches_prueba = Ventas_coches2.toPandas().head(10)


# In[721]:


es = ft.EntitySet(id="name")
es = es.entity_from_dataframe(entity_id="dataprueba", index='Id', dataframe=Ventas_coches_prueba)


# Añadimos una nueva entidad a nuestro EntitySet. Al tener una única tabla esta nueva entidad será la columna por la que queramos agrupar, en nuestro caso: 'Manufacturer'

# In[722]:


es = es.normalize_entity(base_entity_id="dataprueba", new_entity_id="Manufacturer", index="Manufacturer")


# In[723]:


es


# Generamos la matriz nueva de variables y guardamos el schema para añadirlo como atributo después:

# In[724]:


feature_matrix, feats = ft.dfs(entityset=es, target_entity='Manufacturer')


# In[725]:


target_schema = spark.createDataFrame(feature_matrix.reset_index()).schema


# In[726]:


feature_matrix


# Seguimos los mismos pasos que antes recordando que hay que añadir la nueva entidad:

# In[727]:


@pandas_udf(target_schema, PandasUDFType.GROUPED_MAP)   
def generate_features(df_sp):
    es = ft.EntitySet('name', {'dataprueba': (df_sp, 'Id')})
    es = es.normalize_entity(base_entity_id="dataprueba", new_entity_id="Manufacturer", index="Manufacturer")
    return ft.calculate_feature_matrix(feats, es).reset_index()


# Aplicamos la función sobre la columna *Manufacturer* del dataframe de spark que es por la que queremos agregar:

# In[729]:


data_ft = Ventas_coches2.groupby("Manufacturer").apply(generate_features)


# Observamos que el resultado es el esperado:

# In[730]:


data_ft.toPandas()

