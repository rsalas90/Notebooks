
# coding: utf-8

# # SOLUCIÓN ESTÁNDAR EXTRACCIÓN DE FEATURES CON FEATURETOOLS (PYTHON)
# 

# ### OBJETIVO

# El objetivo de este notebook es proporcionar una solución estándar para la creación de variables partiendo de un único dataset.

# ### INTRODUCCION

# Featuretools es la librería que vamos a utilizar. Depende de Deep Feature Synthesis para generar variables. Hay dos tipos de variables primitivas:
# - Transformaciones: aplicadas a una o más columnas de un único dataset
# - Agregaciones: se aplica en varias tablas en entidades con relación padre-hijo, como máximas ventas por cliente.

# In[287]:


import pandas as pd
import numpy as np
import datetime
import featuretools as ft
import featuretools.variable_types as vtypes


# ### Ejemplo 1:

# Supongamos que tenemos un dataset de coches como el que se muestra a continuación. Veamos cuantas variables es capaz de crearnos la librería *featuretools*. 

# In[202]:


#Leemos el dataset
df = pd.read_csv('Automobile_data.csv')


# In[203]:


#Consta de 205 filas y 26 columnas
df.shape


# In[148]:


df.head()


# In[204]:


#Insertamos una nueva columna que será el índice único
df.insert(0,'ID',range(0, (len(df))))


# Antes de utilizar Deep Feature Synthesis es recomendable preparar los datos como un **EntitySet**.  
# En primer lugar, creamos un entityset al que llamaremos *coches*.

# ### PASOS

# In[208]:


#Create new entityset
es = ft.EntitySet(id="coches")


# Ahora debemos añadir entidades. Cada una debe tener un índice (una columna con todos los elementos únicos).

# In[209]:


es = es.entity_from_dataframe(entity_id="coches", index='ID', dataframe=df)


# Como únicamente tenemos un dataset y no podemos establecer relaciones entre varias tablas, añadimos una nueva entidad que será una de nuestras columnas (variables) del dataset previo sobre la cual nos interesa agrupar. En este caso elegimos *make* (la marca del coche) y *bpdy-style*.

# In[210]:


es = es.normalize_entity(base_entity_id="coches", new_entity_id="make", index="make")
es = es.normalize_entity(base_entity_id="coches", new_entity_id="body-style", index="body-style")


# Observamos como se ha creado el EntitySet y se han añadido las endidades así como las relaciones entre ellas (Padre->Hijo). En este caso la variable 'make' está actuando como primary key de la primera tabla.

# In[213]:


es


# In[214]:


feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="make")


# El resultado es un dataframe de nuevas variables para cada marca de coche.

# In[217]:


feature_matrix.head()


# Aquí tendríamos todas las variables que se han creado

# In[220]:


feature_defs


# Si sólo queremos unas variables concretas lo podemos especificar

# In[221]:


feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="make", 
                                 agg_primitives = ['mean'])


# In[222]:


feature_matrix


# ### Ejemplo 2:

# Probemos otro dataset donde tenemos fechas incluídas.

# In[223]:


casas = pd.read_csv('kc_house_data.csv')


# In[224]:


casas.head()


# In[225]:


casas['date'] = pd.to_datetime(casas['date'], format= '%Y%m%dT%H%M%S')


# In[226]:


#Insertamos una nueva columna que será el índice
casas.insert(0,'ID',range(0, (len(casas))))


# Creamos el EntitySet y añadimos el dataset como entidad

# In[253]:


es = ft.EntitySet(id="casas")
es = es.entity_from_dataframe(entity_id="casas", index='ID', dataframe=casas)


# In[263]:


es


# Utilizando el atributo *variables* observamos cómo ha identificado cada columna (tipo)

# In[268]:


es['casas'].variables 


# En ocasiones, *featuretools* considera variables como numéricas cuando son categóricas. Por ello hay que especificarlo a la hora de crear la entidad. Esto ocurre con las variables **waterfront** o **view**, por ejemplo. Para ello, a la hora de crear la entidad hay que ser más explícito:

# In[274]:


variable_types = { 'waterfront': vtypes.Categorical,
      'view': vtypes.Categorical}

es = es.entity_from_dataframe(entity_id="casas", index='ID', dataframe=casas, variable_types=variable_types)


# In[276]:


es['casas'].variables 


# In[277]:


feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="casas")


# In[278]:


len(feature_defs) #se nos han creado 24 features


# In[279]:


feature_defs #Observamos como las últimas variables creadas corresponden separar la fecha en día, mes, año, y dia de la semana.


# In[280]:


feature_matrix.head()


# FeatureTools genera diferentes variables según el tipo de las columnas que tengamos:
# - numéricas: SUM, STD, MAX, SKEW, MIN y MEAN
# - categóricas: NUM_UNIQUE y MODE

# ### ¿Y para el test set?

# Necesitamos aplicar las mismas transformaciones para el conjunto de test. Sin embargo, esto no es obvio.  
# Se aconseja crear un EntitySet usando los datos de test y recalculando las mismas variables llamando a **ft.calculate_feature_matrix** con la lista de variables definidad previamente. Para ello necesitamos codificar esas variables en nuestro conjunto de train y guardar el resultado.

# In[282]:


feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs, include_unknown=False)


# Esto sencillamente LabelBinarizes nuestras variables categóricas. Lo guardamos

# In[286]:


X_train = feature_matrix_enc.copy()
X_train.shape


# A continuación creamos un EntitySet para nuestro conjunto de test.

# In[ ]:


# creating and entity set 'es'
es_tst = ft.EntitySet(id = 'casas')
# adding a dataframe - TEST SET
es_tst.entity_from_dataframe(entity_id = 'casas', dataframe = X_test, index = 'ID')


# A continuación podemos calcular la matriz de variables en nuestro test EntitySet y pasar la lista de variables guardadas de training.

# In[ ]:


feature_matrix_tst = ft.calculate_feature_matrix(features=features_enc, entityset=es_tst)


# ### Use Feature Selection to prune the features
# Una vez hemos generado un gran número de variables nuevas, probablemente necesitamos hacer un proceso de reducción de las mismas. Seguramente muchas estarán altamente correlaciondas por lo que vamos a identificarlas y eliminarlas.

# In[ ]:


# Threshold for removing correlated variables 
threshold = 0.7  

# Absolute value correlation matrix 
corr_matrix = X_train.corr().abs() 
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]
X_train_flt = X_train.drop(columns = collinear_features)
X_test_flt = X_test.drop(columns = collinear_features)
X_train_flt.shape, X_test_flt.shape

