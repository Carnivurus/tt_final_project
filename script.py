import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# # Definir objetivo del proyecto
# - Objetivo: Pronosticar tasa de cancelación de clientes
# - Tipo de modelo: Clasificación
# - Característica objetivo: la columna `'EndDate'` es igual a `'No'`.
# - Métrica principal: AUC-ROC.
# - Métrica adicional: exactitud.
# - Criterios de evaluación:
# - AUC-ROC < 0.75 — 0 SP
# - 0.75 ≤ AUC-ROC < 0.81 — 4 SP
# - 0.81 ≤ AUC-ROC < 0.85 — 4.5 SP
# - 0.85 ≤ AUC-ROC < 0.87 — 5 SP
# - 0.87 ≤ AUC-ROC < 0.88 — 5.5 SP
# - AUC-ROC ≥ 0.88 — 6 SP

# %%
# # Functions
def category_check(df,columns):
    '''When data frame and columns provided, the function
    will return the value counts of each column'''
    for col in columns:
        print()
        print(df[col].value_counts(), end='\n\n')


# %% 
# # Importing datasets
contract_df = pd.read_csv('datasets/contract.csv')
internet_df = pd.read_csv('datasets/internet.csv')
personal_df = pd.read_csv('datasets/personal.csv')
phone_df = pd.read_csv('datasets/phone.csv')

###########
# %% [markdown]
# # Preprocessing

###########
# %% 
# ## CONTRACT
contract_df.head(4)
contract_df.info()
category_check(contract_df,['Type','PaymentMethod'])

# %% [markdown]
# - Se observan columnas con formato PascalCase a excepción de customerID, 
# - Las columnas BeginDate y EndDate pueden ser modificadas al formato timestamp,
# - Lo columna Type puede ser cambiada al tipo Category al ser pocos elementos categóricos (Ocupará menos memoria)
# - PaperlessBilling puede ser una categoría no relevante para el estudio
# - PaymentMethod puede ser cambiada al tipo Category al ser pocos elementos categóricos (Ocupará menos memoria)
# - MonthlyCharges se encuentra correctamente asignado
# - TotalCharges debe ser cambiado a tipo Float


#%%
# Renombrado customerID
contract_df.rename(columns={'customerID': 'CustomerID'}, inplace=True)

# %% 
# Buscando duplicados
contract_df.duplicated().sum()
# No fueron encontrados valores duplicados

# %%
nan_values = contract_df[contract_df.isna().any(axis=1)]
nan_values
# No fueron encontrados valores_nulos

# %%
# Las columnas BeginDate y EndDate pueden ser modificadas al formato timestamp,
contract_df['BeginDate'] = pd.to_datetime(contract_df['BeginDate'], format= '%Y-%m-%d')

# %%
# Cambiando tipos de Columna a category
contract_df['Type'] = contract_df['Type'].astype('category')
contract_df['PaymentMethod'] = contract_df['PaymentMethod'].astype('category')

# %%
# Cambiando Total Charges a float y configurando valores no numéricos a NA

# contract_df[contract_df['TotalCharges'].isna().any()]

valores_no_numericos = pd.to_numeric(contract_df['TotalCharges'], errors='coerce').isna()
# contract_df[valores_no_numericos]
contract_df['TotalCharges'] = pd.to_numeric(contract_df['TotalCharges'], errors='coerce')

# PENDIENTE PARA CONSIDERAR ENCONTRAR TOTAL CHARGES
# %% 
################
# ## INTERNET
internet_df.head(4)
internet_df.info()

category_check(internet_df, internet_df.columns[1:] )

# %% [markdown]
# No hay elementos fuera de los rangos No, Yes.

# %% 
# Buscando duplicados
internet_df.duplicated().sum()

# %% [markdown]
# No fueron encontrados valores duplicados
# Columna customerID será transformada a PascalCase

# %%
#%%
# Renombrado customerID
internet_df.rename(columns={'customerID': 'CustomerID'}, inplace=True)

# %% 
##########
# ## PERSONAL

personal_df.sample(4)
personal_df.info()
category_check(personal_df,personal_df.columns[1:])
personal_df.duplicated().sum()

# %% [markdown]
# - Los elementos no cuentan con valores duplicados ni ausentes, muestra coherencia en los datos.
# - Podrían categorizarse las columnas Gender, Partner y Dependents
# - Gender, CustomerID no estan aplicando CamelCase, se procede a cambiarlos al formato.


# %%
personal_df.rename(columns={'gender': 'Gender'}, inplace=True)
personal_df.rename(columns={'customerID': 'CustomerID'}, inplace=True)


# %% 
##########
# ## PHONE
phone_df.sample(4)
phone_df.info()
category_check(phone_df,phone_df.columns[1:])
phone_df.duplicated().sum()

# %% [markdown]
# - Los elementos no cuentan con valores duplicados ni ausentes, muestra coherencia en los datos.
# - Podrían categorizarse la columna MultipleLines
# - Gender, CustomerID no estan aplicando CamelCase, se procede a cambiarlos al formato.

phone_df.rename(columns={'customerID': 'CustomerID'}, inplace=True)

# %% [markdown]
# # Análisis de caracteristicas

# %%

def plot(df, cols, grid):
    '''Esta funcion crera un grid de todos los elementos deseados a analizar
    en un histograma, solo sera necesario esecificar el df de origen, el nombre de
    las columnas y el grid a utilizar ex. grid=(2,2)'''
    fig, axes = plt.subplots(grid[0],grid[1], figsize=(15,10))
    axes= axes.flatten()

    for i,col in enumerate(cols):
        axes[i].hist(df[col])
        axes[i].set_title(f'{col} Histogram')

    plt.tight_layout()
    plt.show()

def distribution(df):
    '''Esta funcion devuelve los valores únicos en porcentajes
    solo será necesario especificar el df y se analizará cada columna'''
    stats=[]
    for col in df:
        stats.append([df[col].value_counts(normalize=True)*100])
    return stats
# %% [markdown]
# ## Contract dataset

# %%
distribution(contract_df.iloc[:,3:-2])
plot(contract_df, contract_df.iloc[:,3:], grid=(3,2))

# %% [markdown]
# - Tipo de pago: De acuerdo con el gráfico mostrado la preferencia de los usuarios de Interconnect se inclina
# a pagos mensuales como primera opción, seguido de pagos anuales y finalmente pagos cada dos años.
# - Pagos impresos: La mayoría de los usuarios aún prefieren sus facturas en formato físico, aunque observamos que 
# una gran cantidad de usuarios tienen preferencia al formato digital.
# - Metodo de pago: El método de pago mas recurrido es el de cheque electrónico, seguido de cheque por mail,
# transferencia bancaria (automática) y finalmente tarjeta de crédito.

# %%
contract_df['MonthlyCharges'].describe()
plt.boxplot(contract_df['MonthlyCharges'],showfliers=True)
plt.show()

contract_df['TotalCharges'].describe()
sns.boxplot(contract_df['TotalCharges'],showfliers=True)
plt.show()

# %% [markdown]
# - Cargo Mensual: Los cargos mensuales tienen una media alrededor de 65 y no presenta valores atípicos,
# además, presenta una mínima de 18.25 y una máxima de 118.75 M/N
# - Cargos totales: Los cargos totales tienen una media de 2283.30 con una maxima de 8684.80 y una mínima de 18.8,
# no se presentan valores atípicos.

# %% [markdown]
# ## Internet dataset

# %%
distribution(internet_df.iloc[:,1:])
plot(internet_df, internet_df.iloc[:,1:], grid=(4,2))

# Obtener la distribución de los valores.

# %% [markdown]

# - Servicio de internet: La fibra óptica es la infraestructura mas utilizada para el servicio de Interconnect, 
# sin embargo, mas de 2000 usuarios (alrededor del 43%) aún cuentan con DSL lo cuál no podría ser la mejor opción con las necesidades actuales.
# - Servicio de seguridad online: La mayoría de los usuarios no usan este servicio, solo un 37% lo tiene contratado, esto puede deberse a
# a la falta de necesidad por los servicios integrados de seguridad por windows/apple.
# - Servicio de respaldo: Un 44% de los usuarios tienen contratado este servicio para respaladar su información
# - Servicio de protección de dispositivos: Un 44% de los usuarios hacen uso de esta servicio.
# - Servicio de Tech Support: El 63% no utiliza el servicio de soporte técnico
# - Streaming TV: Casi la mitad de los usuarios (49%) utilizan el servicio de streaming TV
# - StreamingMovies: La mitad de los usuarios utilizan el servicio de streaming TV

# %% [markdown]
# ## Personal dataset

plot(personal_df, personal_df.iloc[:,1:], grid=(2,2))
distribution(personal_df.iloc[:,1:])

# %% [markdown]
# - Genero: La distribución de genero es equitativa, 50% para ambos grupos.
# - SeniorCitizen: El 83% de usuarios son mayores de 60 años.
# - Compañero: El 51% de usuarios tienen una pareja
# - Dependientes: Solo el 30% de los usuarios tienen dependientes.


# %% [markdown]
# ## Phone dataset

# %% 
distribution(phone_df.iloc[:-1])
plt.hist(phone_df['MultipleLines'])
plt.show()


# # %% [markdown]
# - Multiples Lineas: El 53% de los usuarios posen una linea, mientras que el 47% tienen contratadas multiples lineas.


contract_df.sample(1)
internet_df.sample(1)
personal_df.sample(1)
phone_df.sample(1)

merged_df = pd.merge(contract_df, internet_df, how='outer', on='CustomerID')
merged_df = pd.merge(merged_df, personal_df, how='outer', on='CustomerID')
merged_df = pd.merge(merged_df, phone_df, how='outer', on='CustomerID')
merged_df.info()
merged_df[merged_df.isnull().any(axis=1)]


# %% [markdown]
# Tenemos valores faltantes al realizar el merge entre desde los datasets internet_df y phone_df, sera necesario imputar
# los datos o removerlos.
# - Contamos con 7043 datos totales con valores nulos.

# %% 
merged_df.info()
impute_df = merged_df.copy()

## Analizando cuales pasar a numéricos

object_cols= impute_df.select_dtypes(exclude=['number'])
non_objects_cols = impute_df.select_dtypes(include=['number'])

object_cols.info()

from sklearn.preprocessing import LabelEncoder
def label_encoder(df):
    label_mappings ={}
    for col in df:
        le = LabelEncoder()
        df[col]= le.fit_transform(df[col])
        label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    return df, label_mappings

def reverse_label_encoding(df, mappings):
    df_reversed = df.copy()
    for col, mapping in mappings.items():
        reverse_mapping = {v: k for k, v in mapping.items()}
        df_reversed[col] = df_reversed[col].map(reverse_mapping)
    return df_reversed


df_encoded, label_mappings = label_encoder(object_cols.iloc[:,3:])

# Usar al final
# df_original = reverse_label_encoding(df_encoded, mappings)
reverse_label_encoding(df_encoded, label_mappings)

## LA CORRELACION NO SE PUEDE APLICAR SI LOS VALORES NO SON NUMERICOS.

new_df = pd.concat([object_cols.iloc[:,0:3], df_encoded, non_objects_cols], axis=1)

# Definiendo correlación ppara valores no nulos.
correlation_df = new_df[~new_df.isna().any(axis=1)]
corr = correlation_df.iloc[:,3:].corr()
print(corr)

# Graficando la correlación
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# %% [markdown]
# No encontramos una correlación clara en las variables.


from sklearn.impute import SimpleImputer

# First Option: using SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
impute_df = pd.DataFrame(imp.fit_transform(impute_df), columns=merged_df.columns)
impute_df.info()
impute_df.iloc[7032,:]

# Second Option: removing null rows
removed_nulls = merged_df.copy()
removed_nulls = removed_nulls.dropna()
removed_nulls.info()

# Third Option: using machine learning

new_df

not_na_df = new_df[~new_df.isna().any(axis=1)]
na_df = new_df[new_df.isna().any(axis=1)]

not_na_df.shape
na_df.shape

# Defining features and objective
x_train = not_na_df.drop(columns='TotalCharges')
y_train = not_na_df['TotalCharges']

rows = na_df.index
x_test = na_df.drop(columns='TotalCharges')






from sklearn.linear_model import LogisticRegression

#ME DETUVO YA QUE ES NECESARIO TENER TODOS LOS VALORES EN NUMÉRICOS, NO CATEGÓRIGOS

lr = LogisticRegression()
# split
lr.fit(x_train,y_train)
target_score = lr.score(x_train,y_train)
pred = lr.predict()
