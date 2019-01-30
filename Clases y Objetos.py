
# coding: utf-8

# ### CLASES

# Las **clases** son los modelos sobre los cuáles se construirán nuestros objetos. Podemos tomar como ejemplo de clases, el gráfico que hicimos en la página 8 de este documento.
# 
# En Python, una clase se define con la instrucción *class* seguida de un nombre genérico para el objeto

# Las **propiedades** son las características intrínsecas del objeto. Éstas, se representan a modo de variables, solo que técnicamente, pasan a denominarse propiedades.

# Los **métodos** son funciones (como las que vimos en el capítulo anterior), solo que técnicamente se denominan métodos, y representan acciones propias que puede realizar el objeto (y no otro).

# Pensemos a las clases e instancias como planos y casas. Cuando diseñamos un plano, incluimos en él información tal como medida de la vivienda, números de habitaciones y baños y su correspondiente ubicación, pisos de este o aquel material, distribución de los espacios, etc. El plano no es en sí mismo ninguna casa; antes bien, describe cómo se verá ésta una vez construida. De modo que, si el plano contiene una puerta lateral, podemos asegurar que no nos encontraremos con una pared cuando nos dirijamos allí en la casa actual.
# 
# De la misma forma, una clase no constituye ningún objeto en particular. Más bien define las propiedades que tendrán las instancias de dicha clase una vez creadas; y aquí por propiedades entiendo variables y funciones. Las variables y funciones que están dentro de una clase llevan el nombre de atributos y métodos, respectivamente.

# In[1]:


class Ojo():
    forma = "pequeño" 
    color = "miel" 
    tamaño = "achinado"
    
class Pelo():
    color = "castaño" 
    textura = "suave" 

class Objeto():
    color = "verde"
    tamaño = 'grande'
    aspecto = "feo"
    ojos = Ojo()
    pelo = Pelo()
    
    def Flotar(self):  #Notar que el primer parámetro de un método, siempre debe ser self
        print (12)
        


# ### OBJETOS

# Las clases por sí mismas, no son más que modelos que nos servirán para crear objetos en concreto. Podemos decir que una clase, es el razonamiento abstracto de un objeto, mientras que el objeto, es su materialización. A la acción de crear objetos, se la denomina instanciar una clase y dicha instancia, consiste en asignar la clase, como valor a una variable:

# In[11]:


et = Objeto()


# In[12]:


et.color


# In[13]:


et.tamaño


# In[14]:


et.color = "rosa" 
print (et.color)


# Algunos objetos comparten las mismas propiedades y métodos que otro objeto, y además agregan nuevas propiedades y métodos. A esto se lo denomina herencia: una clase que hereda de otra. Vale aclarar, que en Python, cuando una clase no hereda de ninguna otra, debe hacerse heredar de object, que es la clase principal de Python, que define un objeto.

# In[24]:


class Dedo(object): 
    longitud = "10 mm" 
    forma = "gordo" 
    color = "carne"

class Pie(object): 
    forma = "" 
    color = "" 
    dedos = Dedo() 
    
class NuevoObjeto(Objeto): 
    dedo = Dedo() 
 
    def saltar(self): 
        print ("Ha saltado!")


# #### Acceder a los métodos y propiedades

# Una vez creado un objeto, es decir, una vez hecha la instancia de clase, es posible acceder a su métodos y propiedades. Para ello, Python utiliza una sintaxis muy simple: el nombre del objeto, seguido de punto y la propiedad o método al cuál se desea acceder:

# In[26]:


objeto = Dedo() 
print (objeto.forma) 


# In[29]:


variable = NuevoObjeto.saltar()
print (variable)
# print (NuevoObjeto.otro_metodo())


# ### Ejemplo

#  Supongamos que necesitamos desarrollar una aplicación para gestionar ─agregar, remover, editar─ la información de los estudiantes en un instituto.

# In[31]:


class Student():
    pass


# In[39]:


#Creo una instancia del objeto. (A los objetos también se los llama instancias)
student = Student()


# In[40]:


student.name = "Pablo"


# In[41]:


student.subjects = [5, 14, 3]


# In[42]:


student2 = Student()
student2.name = "Pedro"
student2.subjects = [1, 9]


# Este abordaje tiene un problema. En primer lugar está la obvia repetición de código. Si quisiéramos añadir un atributo o bien modificar el nombre de uno existente, deberíamos hurgar en todo el código en busca de ocurrencias. Por otro lado, ¿qué pasaría si quisiéramos chequear que el atributo name sea una cadena y el atributo subjects una lista? Pues bien, simplemente, digamos, creemos una función que se encargue de ello.

# In[52]:


class Student:
    pass
    def initialize(student, name, subjects):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        elif not isinstance(subjects, list):
            raise TypeError("subjects must be a list")
        student.name = name
        student.subjects = subjects

    
student = Student()
Student.initialize(student, "Pablo", [5, 14, 3])
student2 = Student()
Student.initialize(student2, "Pedro", [1, 9])


# Initialize() es un método de la clase Student cuyo primer argumento siempre es una instancia de ésta, Python nos permite llamarlo usando la nomenclatura instancia.metodo(...) como abreviación para Clase.metodo(instancia, ...). Por esta razón, el código anterior lo reescribimos como sigue.

# In[53]:


student = Student()
student.initialize("Pablo", [5, 14, 3])
student2 = Student()
student2.initialize("Pedro", [1, 9])


# La creación de un método para inicializar atributos es una práctica tan común en la programación orientada a objetos, que el lenguaje reserva un nombre especial para esa función: __init__(). En Python, los objetos que comienzan y terminan con doble guión bajo siempre tienen un funcionamiento en particular y no son invocados directamente. Así, con el renombre pautado, el código de la clase pasaría a ser:

# In[56]:


class Student:
    def __init__(student, name, subjects):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        elif not isinstance(subjects, list):
            raise TypeError("subjects must be a list")
        student.name = name
        student.subjects = subjects
        
    def print_info(student):
        print("Nombre: {}.".format(student.name))
        print("Asignaturas: {}.".format(student.subjects))


# No obstante, como dijimos, **_ _init_ _()** no está pensado para ser llamado directamente. Antes bien, Python lo invoca de forma automática cuando creamos una instancia de la clase. Y los argumentos que le pasamos a la clase son pasados al método de inicialización. De modo que nuestro código posterior se reduce a lo siguiente.

# In[58]:


student = Student("Pablo", [5, 14, 3])
student2 = Student("Pedro", [1, 9])


# In[59]:


student.print_info()


# ### ¿Qué es SELF?

# Todos los métodos dentro de una clase llevan por primer argumento una instancia de dicha clase, sobre la cual se efectúan los cambios. Por cuanto nuestra clase Student representa un estudiante, hemos nombrado a este primer argumento student. No obstante, por convención, se lo llama self ─es decir, «yo»─.

# In[60]:


class Student:
    def __init__(self, name, subjects):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        elif not isinstance(subjects, list):
            raise TypeError("subjects must be a list")
        self.name = name
        self.subjects = subjects
    
    def print_info(self):
        print("Nombre: {}.".format(self.name))
        print("Asignaturas: {}.".format(self.subjects))


# Por esta causa self no es ningún objeto mágico o palabra reservada. Sencillamente es una instancia de la clase desde la cual está siendo invocado. Recordemos que ambas llamadas a continuación son equivalentes.

# In[61]:


Student.print_info(student)
student.print_info()


# Así, considerando el siguiente código:

# In[62]:


student.print_info()
student2.print_info()


# En la primera llamada, self == student, mientras que en la segunda self == student2.

# ### Encapsulamiento

# En Python todos los métodos y atributos son públicos. Públicos, entendidos en la terminología de otros lenguajes orientados a objetos como Java o C++ que distinguen entre elementos públicos y privados. No obstante, como convención, se prefija un guión bajo a aquellos métodos o atributos que quieran ser catalogados como privados. En otras palabras, un guión bajo delante del nombre de un objeto indica que éste no debería ser utilizado desde fuera de la clase.
# 
# 

# Por ejemplo, si bien hemos chequeado en la inicialización de Student que el atributo name sea una cadena, nada nos impide alterarlo luego por fuera de la clase.
# 
# 

# In[64]:


student = Student("Pablo", [5, 14, 3])
student.name = 1  # No es una cadena.


# Sería útil, por ende, crear los métodos set_name() y get_name() para hacer las comprobaciones pertinentes.

# In[65]:


class Student:
    def __init__(self, name, subjects):
        self.set_name(name)
        if not isinstance(subjects, list):
            raise TypeError("subjects must be a list")
        self.subjects = subjects
    
    def set_name(self, name):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self._name = name
    
    def get_name(self):
        return self._name


# In[73]:


student = Student("Pablo", [5, 14, 3])


# In[76]:


student.set_name(1)  # TypeError.

