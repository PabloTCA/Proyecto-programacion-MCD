# Trabajo Final de Programación para los cursos de la Maestría en Ciencia de Datos de la Universidad de Sonora
![Universidad de Sonora](https://investigadores.unison.mx/skin/headerImage/)

## Introducción

En este trabajo, he realizado diversos análisis basados en los requisitos solicitados. Mi nombre es Pablo Trinidad Chávez Amavizca y he sido responsable de llevar a cabo los análisis y observaciones presentados a lo largo de este trabajo.

Los datos utilizados en este proyecto corresponden a los Índices de Marginación del año 2020, los cuales son de libre acceso y se encuentran disponibles en la página oficial del Gobierno de México. Se realizó un proceso de limpieza y preparación de los datos para garantizar su correcta utilización en el programa.

Sin más preámbulos, espero que este trabajo sea de su agrado y refleje mi dedicación y compromiso con el campo de la Ciencia de Datos.


Este trabajo presenta un análisis detallado de los Índices de Marginación en México. Los datos utilizados en este proyecto fueron recopilados de diversas fuentes confiables y están disponibles para su acceso público en la página oficial del Gobierno de México.

## Objetivo

El objetivo de este análisis es examinar los índices de marginación en diferentes estados y municipios de México, con el fin de identificar patrones, tendencias y posibles relaciones entre variables.

## Proceso de Análisis

1. Recopilación de Datos: Se obtuvieron los datos de los Índices de Marginación del año 2020 desde la página oficial del Gobierno de México.

2. Limpieza y Preparación de Datos: Se realizó un proceso de limpieza de datos para eliminar valores nulos, corregir errores y estandarizar formatos. Además, se realizó una adecuada preparación de los datos para su posterior análisis.

3. Análisis Exploratorio: Se realizaron diversos análisis exploratorios para examinar la distribución de los índices de marginación, identificar valores atípicos y visualizar relaciones entre variables.

4. Generación de Gráficos: Se crearon gráficos y visualizaciones para representar los resultados del análisis de manera clara y comprensible.

5. Interpretación de Resultados: Se realizaron observaciones y conclusiones basadas en los análisis realizados, buscando identificar patrones y tendencias relevantes en los índices de marginación.



### Importación de Librerías y Conversión de Datos

En primer lugar, se realizará la importación de las librerías necesarias para llevar a cabo el análisis. En este caso, se utilizarán las librerías numpy, pandas y matplotlib. A continuación, se ejecutan los comandos correspondientes para asegurarse de que estas librerías estén disponibles en el entorno de trabajo.

Seguidamente, se procederá a la conversión del archivo descargado en formato Excel a un archivo CSV. Aunque Python puede leer archivos Excel directamente, se optará por esta conversión para evitar posibles incompatibilidades y asegurar la correcta lectura de los datos. El archivo CSV resultante será guardado en la misma ubicación.

Una vez realizada la conversión, se procederá a leer el archivo CSV y almacenar los datos en una variable denominada "indice". A continuación, se mostrará un avance de los datos para verificar su correcta importación.

### Código

```python
# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Conversión del archivo a formato CSV
ruta_excel = 'C:\\Users\\Hatok\\Downloads\\IMM2020.xls'   # Leemos el archivo para poder convertirlo a CSV
dt = pd.read_excel(ruta_excel)                            # Lo pondremos en un DataFrame para hacer la convercion
ruta_csv = 'C:\\Users\\Hatok\\Downloads\\IMM2020.csv'     # Elegimos la ruta de guardado de nuestro nuevo CSV
dt.to_csv(ruta_csv, index=False)                          # Hacemos el cambio

# Lectura de los datos en un DataFrame
indice = pd.read_csv('C:\\Users\\Hatok\\Documents\\Libretas Jupyter\\IMM2020.csv')

# Avance de los datos importados
indice.head(10) #Vamos a tener que hacer una limpieza en este archivo, tenemos unas filas que no nos sirven.

```
Con este código, se asegura la correcta preparación de los datos y se está listo para proceder con el análisis y las visualizaciones pertinentes.

## Desarrollo del Análisis

### Limpieza de Datos

A continuación, se procederá a realizar la limpieza de los datos. En primer lugar, se eliminarán las primeras dos filas, ya que contienen valores irrelevantes para el análisis. Posteriormente, se eliminarán las filas con los índices 3, 4, 5 y 6.

```python
# Eliminación de las primeras dos filas
indice = indice.iloc[2:]
indice.head(5)

# Eliminación de filas con índices 3, 4, 5 y 6
indice = indice.drop([3, 4, 5, 6])
indice.reset_index(drop=True, inplace=True)
indice.head()
```

Una vez realizada la eliminación de filas innecesarias, se procederá a reemplazar los nombres de las columnas actuales con los nombres correspondientes. Se utilizará la fila con el índice 2, que contiene los nombres actualizados de las columnas.

```python
# Reemplazo de los nombres de las columnas
new_columns = indice.iloc[0]
indice = indice.iloc[1:]
indice.columns = new_columns
indice.head(2)
```

Para asegurar la consistencia del dataframe, se reiniciarán los índices.

```python
# Reinicio de los índices
indice.reset_index(drop=True, inplace=True)
indice
```

Además, se observó la presencia de dos filas vacías al final del dataframe, las cuales se eliminaron para evitar problemas futuros.

Con estas acciones de limpieza y preparación, el dataframe queda listo para su análisis y visualización posterior.



```python
# Tambien tenemos dos valores en la parte inferior del dataframe que no necesitamos y solo nos estan afectando
indice = indice.iloc[:-2]
indice.describe()

```

Con la descripción dada realice las siguientes observaciones

## Observaciones del índice de marginación del 2020

A continuación, se presentan algunas observaciones destacadas sobre el índice de marginación del año 2020:

- **Distribución de municipios**: A simple vista, se destaca que el estado de Oaxaca cuenta con la mayor cantidad de municipios en comparación con otros estados. Aunque este dato puede no ser relevante en sí mismo, resulta curioso. También llama la atención que existen siete municipios llamados Benito Juárez, y uno de ellos se encuentra en el estado de Sonora. Además, se ha identificado que al menos en el año 2020, hay dos municipios con una población exactamente del mismo tamaño.

- **Relación entre analfabetismo y falta de educación básica**: Se ha detectado una coincidencia interesante en dos lugares donde el porcentaje de población con 15 años o más que es analfabeta es de 8.33%. Al mismo tiempo, se observa que el porcentaje de población de 15 años o más sin educación básica es del 40%. Sería interesante investigar si existe una relación entre estos dos indicadores. ¿Será que dentro de ese 40% de población sin educación básica, un 8% corresponde a personas analfabetas?

- **Población con ingresos menores a 2 salarios mínimos**: Se ha notado que en cuatro lugares el porcentaje de población ocupada con ingresos inferiores a 2 salarios mínimos es del 94%. Aunque la frecuencia de esta situación no es alta, resulta significativo que en estos cuatro lugares casi toda su población se encuentre ganando menos de 2 salarios mínimos. Sería interesante examinar si existe alguna relación entre estos porcentajes y los niveles de población sin educación básica.

Estas observaciones iniciales brindan pistas interesantes sobre los datos del índice de marginación del 2020 y podrían ser puntos de partida para realizar análisis más detallados y descubrir posibles relaciones entre los diferentes indicadores.




## Gráfica del Porcentaje de Marginación de Municipios por Estado

A continuación, se presenta el código para generar una gráfica de barras que muestra el porcentaje de marginación de los municipios por estado. Se utiliza un ciclo `for` para graficar los diferentes grados de marginación en cada iteración.

```python
import matplotlib.pyplot as plt
import numpy as np

# Agrupar los datos por estado y grado de marginación y calcular el porcentaje
m_estados = indice.groupby(['Nombre de la entidad', 'Grado de marginación, 2020']).size().unstack()
porcentaje = m_estados.apply(lambda x: x / x.sum() * 100, axis=1)

ancho_barra = 0.2
espaciado_barra = 0.1
posiciones = np.arange(len(porcentaje.index))

marginacion_deseada = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']
fig, ax = plt.subplots(figsize=(15, 6))

for i, marginacion in enumerate(marginacion_deseada):
    desplazamiento = (ancho_barra + espaciado_barra) * i
    barras = ax.bar(posiciones + i * ancho_barra, porcentaje[marginacion], ancho_barra, label=marginacion)

ax.set_xticks(posiciones)
ax.set_xticklabels(porcentaje.index, rotation='vertical')
ax.set_xlabel('Estados')
ax.set_ylabel('Porcentaje de Municipios')
ax.set_title('Porcentaje de Marginación de Municipios por Estado')
ax.legend(title='Grado de Marginación', bbox_to_anchor=(1, 1))
ax.set_xlim(-0.5, len(posiciones) - 1.5)

plt.savefig('Grados_de_marginacion_por_municipio.png', dpi=600, bbox_inches='tight')  # Guardar la gráfica
plt.show()
```
![Grados_de_marginacion_por_municipio](https://github.com/PabloTCA/Proyecto-programacion-MCD/assets/135001016/497c2db6-fbd8-4471-98e9-a4cc32b60d16)



## Gráfica del Porcentaje de Marginación de Población por Estado

A continuación, se presenta una adaptación del código anterior para generar una gráfica de barras que muestra el porcentaje de marginación de la población por estado. Se utilizan los mismos principios, pero con modificaciones en las variables para reflejar el cambio de enfoque.



```python
poblacion_por_estado=indice.groupby(['Nombre de la entidad','Grado de marginación, 2020'])['Población total'].sum().unstack()
poblacion_porcentaje = poblacion_por_estado.div(poblacion_por_estado.sum(axis=1), axis=0)


ancho_barra = 0.2
espaciado_barra = 0.1
posiciones = np.arange(len(poblacion_porcentaje.index))

marginacion_deseada = ['Muy bajo','Bajo','Medio','Alto','Muy alto']
fig, ax = plt.subplots(figsize=(15,6))

for i, marginacion in enumerate(marginacion_deseada):
    desplazamiento = (ancho_barra + espaciado_barra)* i
    barras = ax.bar(posiciones + i * ancho_barra, poblacion_porcentaje[marginacion], ancho_barra, label=marginacion)

ax.set_xticks(posiciones)
ax.set_xticklabels(poblacion_porcentaje.index, rotation='vertical')


ax.set_xlabel('Estados')
ax.set_ylabel('Porcentaje de la poblacion')
ax.set_title('Porcentaje de marginacion de poblacion total por estado')
ax.legend(title='Grado de marginacion', bbox_to_anchor=(1,1) )

ax.set_xlim(-0.5, len(posiciones) - 1.5)
plt.tight_layout()

plt.savefig('Porcentaje de marginacion por poblacion total de estados.jpg', dpi=600,bbox_inches='tight')
plt.show()

```
![Porcentaje de marginacion por poblacion total de estados](https://github.com/PabloTCA/Proyecto-programacion-MCD/assets/135001016/64dc2270-e9ca-4cb4-bb08-2646f6ef926f)


# Comparativa de gráficas

En el análisis de las dos gráficas, una que muestra los porcentajes de marginación de los municipios por estado y otra que muestra el porcentaje de marginación de la población total por estado, se han identificado las siguientes observaciones:

- **Variación en grados de marginación**: Aunque algunos estados presentan municipios con altos grados de marginación, esto no necesariamente significa que la mayoría de la población se encuentre en esos mismos grados de marginación. Por el contrario, parece ser que dichos grados de marginación elevados están asociados con áreas de baja densidad poblacional que se encuentran altamente marginadas. Por otro lado, en los municipios con mayor concentración de población, el índice de marginación tiende a disminuir. Esta observación sugiere una posible relación: "A menor densidad poblacional, mayor es el índice de marginación". Sería necesario realizar un análisis más exhaustivo para confirmar definitivamente si existe una relación entre estas dos categorías.

- **Caso de Oaxaca**: Al analizar el estado de Oaxaca en función de su población total, se observa que los cinco índices de marginación se encuentran muy nivelados entre sí. Aunque el índice de marginación alto es ligeramente mayor, no presenta una diferencia significativa. Sin embargo, al examinar el resumen del dataframe, se destaca que Oaxaca también es el estado con la mayor cantidad de municipios. Al revisar la gráfica de marginación por municipios, se aprecia una amplia diferencia entre aquellos municipios con un índice de marginación muy bajo (alrededor del 10%) y aquellos con un índice de marginación alto (superior al 40%). Esto sugiere que más del 40% de los municipios en Oaxaca presentan un grado de marginación alto.

Estas observaciones iniciales revelan posibles patrones y relaciones entre los grados de marginación, la densidad poblacional y la distribución de la marginación en diferentes municipios y estados. Sería recomendable profundizar en el análisis para comprender mejor estas relaciones y sus implicaciones.



## Gráfica de Dispersión (Scatterplot) del Porcentaje de Analfabetismo por Municipio

A continuación, se utilizará un gráfico de dispersión (scatterplot) para visualizar la relación entre el porcentaje de analfabetismo y los municipios. En este tipo de gráfico, se representará cada municipio como un punto en el plano, donde el eje x corresponde al porcentaje de analfabetismo y el eje y representa la ubicación del municipio.



```python
indice.loc[:,'Población total'] = pd.to_numeric(indice['Población total'], errors='coerce')
indice.loc[:,'% Población de 15 años o más analfabeta '] = pd.to_numeric(indice['% Población de 15 años o más analfabeta '], errors='coerce')

df_filtrado = indice[indice['Población total'] < 5000]

# Obtener los porcentajes de analfabetismo y población en localidades de menos de 5,000 habitantes
porcentaje_analfabetismo = df_filtrado['% Población de 15 años o más analfabeta ']
porcentaje_poblacion = df_filtrado['Población total']

# Crear el gráfico de dispersión
plt.scatter(porcentaje_poblacion, porcentaje_analfabetismo)
plt.xlabel('Población')
plt.ylabel('Porcentaje de Analfabetismo')
plt.title('Relación entre Porcentaje de Analfabetismo y Población')
plt.show()

```
![Porcentaje de analfabetismo](https://github.com/PabloTCA/Proyecto-programacion-MCD/assets/135001016/fe4dfaac-38b0-4c2a-b94d-d25c72e6ea1b)

# Observaciones

Al analizar la gráfica, se pueden hacer las siguientes observaciones:

- **Aumento en la densidad de puntos**: Se puede apreciar que a medida que disminuye la densidad poblacional de los municipios, hay un aumento en la densidad de puntos en la gráfica. Esto sugiere que en los municipios con menor densidad poblacional, se encuentran más casos de analfabetismo. Aunque los porcentajes de analfabetismo no son muy altos en estos municipios, se puede notar una tendencia de incremento a medida que la densidad poblacional disminuye.

- **Cluster denso de puntos**: Existe un cluster más denso de puntos en la gráfica que se encuentra en el rango de 0% a 10% de analfabetismo. Esta concentración de municipios con bajos porcentajes de analfabetismo se encuentra espaciada entre 500 habitantes y 2000 habitantes. Sin embargo, es importante tener en cuenta que esto podría ser simplemente una coincidencia y no necesariamente indicar una relación directa. Sería necesario realizar un análisis más detallado para determinar si existe una relación significativa en esta zona específica.

Estas observaciones iniciales sugieren una posible relación entre la densidad poblacional, el porcentaje de analfabetismo y la distribución de los municipios en la gráfica. Para una comprensión más precisa, se recomienda realizar un análisis más profundo y considerar otros factores relevantes.


# Desarrollando un nuevo dataframe


En esta sección, se procedió a crear un nuevo dataframe basado en columnas específicas de interés, centrándose en la educación y el salario mínimo. Después de realizar una observación exhaustiva, se seleccionaron las columnas que se consideraron más relevantes.

```python
# Desarrollando un nuevo dataframe
df_nuevo = indice.copy()
agre = {
    '% Viviendas particulares con hacinamiento': 'sum',
    '% Ocupantes en viviendas particulares sin drenaje ni excusado': 'sum',
    '% Ocupantes en viviendas particulares sin energía eléctrica': 'sum',
    '% Ocupantes en viviendas particulares con piso de tierra': 'sum',
    '% Ocupantes en viviendas particulares sin agua entubada': 'sum',
    '% Población de 15 años o más sin educación básica': 'sum',
    '% Población ocupada con ingresos menores a 2 salarios mínimos': 'sum',
    'Grado de marginación, 2020': 'sum'
}

df_analisis = df_nuevo.groupby(['Nombre de la entidad', 'Nombre del municipio']).agg(agre)

df_analisis.max()
```

A pesar de los intentos iniciales de graficar utilizando scatter plots con la biblioteca matplotlib, no se logró obtener los resultados deseados. En consecuencia, se decidió utilizar la librería plotly, que ofrece gráficas interactivas visualmente atractivas. En este caso particular, se graficó el porcentaje de población de 15 años o más sin educación básica en relación con el porcentaje de población con ingresos inferiores a dos salarios mínimos.
Lo primero que hice para ver como acomodar un nuevo dataframe es poner las columnas que mas me llamaron la atencion y observar que valores podrian resultar mas interesantes, despues de un buen tiempo de observacion me decidi por elegir la educacion y el salario minimo.


```python
import plotly.express as px

df_analisis
df_sorted = df_analisis.sort_values('% Población  de 15 años o más sin educación básica')
municipios = df_sorted.index.get_level_values('Nombre del municipio')

df_sorted['Nombre del municipio'] = municipios

fig = px.scatter(df_sorted, x = '% Población  de 15 años o más sin educación básica', y = '% Población ocupada con ingresos menores a 2 salarios mínimos', hover_name='Nombre del municipio')

fig.update_layout(
    title='Relación entre educación y salario mínimo por municipio',
    xaxis=dict(
        title='% Población de 15 años o más sin educación básica',
        tickformat='%'
    ),
    yaxis=dict(
        title='% Población ocupada con ingresos menores a 2 salarios mínimos',
        tickformat='%'
    ),
    hoverlabel=dict(
        namelength=-1
    )
)

fig.show()

```

![image](https://github.com/PabloTCA/Proyecto-programacion-MCD/assets/135001016/f752d4b4-e19b-4af3-91e9-883e2252f4cd)



# Graficando la relación entre educación y el grado de marginación

A continuación, se utiliza el mismo código utilizado para la primera gráfica, pero ahora se compara la educación con el grado de marginación.

```python
municipios = df_sorted.index.get_level_values('Nombre del municipio')

df_sorted['Nombre del municipio'] = municipios

fig = px.scatter(df_sorted, x='% Población de 15 años o más sin educación básica', y='Grado de marginación, 2020', hover_name='Nombre del municipio')

fig.update_layout(
    title='Relación entre educación y el grado de marginación',
    xaxis=dict(
        title='% Población de 15 años o más sin educación básica',
        tickformat='%'
    ),
    yaxis=dict(
        title='Grado de marginación',
        tickformat='%'
    ),
    hoverlabel=dict(
        namelength=-1
    )
)

fig.show()
```

![image](https://github.com/PabloTCA/Proyecto-programacion-MCD/assets/135001016/30749567-914e-4376-a5df-3e2f2be62c9f)


La gráfica generada muestra la relación entre el porcentaje de población de 15 años o más sin educación básica y el grado de marginación. Cada punto en la gráfica representa un municipio, y al pasar el cursor sobre los puntos se mostrará el nombre del municipio correspondiente.

Antes de proceder con el análisis, guardaré este nuevo dataframe en un archivo Parquet. A continuación, se muestra el código utilizado:

```python
df3 = indice.copy()
agre={'% Población  de 15 años o más sin educación básica':'sum','% Población ocupada con ingresos menores a 2 salarios mínimos':'sum',
      'Grado de marginación, 2020':'sum'
      }

df_educacion = df3.groupby(['Nombre de la entidad', 'Nombre del municipio']).agg(agre)
df_educacion.to_parquet('C:\\Users\\Hatok\\Documents\\educaciondataframe.parquet')
```

En este código, se crea un nuevo dataframe llamado `df_educacion` basado en el dataframe `df3`. Luego, se realiza un agrupamiento y agregación de los datos utilizando las columnas "Nombre de la entidad" y "Nombre del municipio". Finalmente, el dataframe resultante se guarda en un archivo Parquet con el nombre "educaciondataframe.parquet" en la ubicación especificada "C:\Users\Hatok\Documents\".

Es importante tener en cuenta que se debe ajustar la ruta de guardado del archivo Parquet según la ubicación deseada en el sistema.


# Análisis

Realicé un análisis para investigar si existe una fuerte correlación entre el porcentaje de población sin educación básica y el porcentaje de población con ingresos menores a 2 salarios mínimos. Mi suposición inicial era que podría haber una correlación, pero no esperaba que fuera tan marcada como se observa en las gráficas.

Al analizar las dos gráficas, queda claro que la educación es un factor clave en la problemática de la marginación. Se puede observar una pendiente ascendente en la relación entre el porcentaje de personas sin educación básica y el porcentaje de marginación. Esto indica que a medida que aumenta el porcentaje de personas sin educación básica, también se incrementa el grado de marginación.

Estos hallazgos son evidencia contundente de que la educación desempeña un papel fundamental en el desarrollo y la calidad de vida de las comunidades. Es crucial realizar un análisis más profundo para comprender las situaciones en las áreas con menor educación, y evaluar si existen dificultades de acceso o barreras que impiden el acceso a la educación.

Además, se deben buscar soluciones a corto plazo para brindar ayuda inmediata a estas comunidades mientras se desarrollan estrategias a largo plazo. Es importante trabajar en un cambio generacional, enfocándose en mejorar la situación educativa para lograr un aumento en la calidad de vida de los habitantes de estas áreas marginadas. Este enfoque a largo plazo puede resultar beneficioso para el desarrollo integral de estas zonas, generando un impacto positivo en la calidad de vida de sus habitantes.



## Conclusiones

A través de este trabajo, se ha podido obtener una mejor comprensión de los índices de marginación en México. Se han identificado patrones interesantes, tendencias y posibles relaciones entre variables, lo que nos permite tener una visión más clara de la situación socioeconómica en diferentes estados y municipios.

Es importante destacar que este análisis es solo una aproximación inicial y que se requiere un análisis más profundo y detallado para obtener conclusiones más precisas. No obstante, este trabajo proporciona una base sólida para futuras investigaciones y acciones encaminadas a abordar los desafíos asociados a la marginación en México.

Los resultados presentados en este trabajo demuestran la importancia de utilizar datos abiertos y accesibles para realizar análisis rigurosos y fundamentados. La transparencia y disponibilidad de los datos gubernamentales son fundamentales para el desarrollo de soluciones efectivas y la toma de decisiones informadas.

## Referencias

- Gobierno de México: [Página oficial de los Índices de Marginación](https://www.gob.mx/cms/uploads/attachment/file/539367/Indice_Marginacio_n_2020_Base_de_Datos.xlsx)
