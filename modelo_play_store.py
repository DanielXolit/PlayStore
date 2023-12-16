
## Hola UDD

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

# Cargamos los datos:
df = pd.read_csv('googleplaystore.csv')
df.head()

df = df[['Rating','Reviews','Price']]

df.info()

#cambio la columna de tipo object a int64 en Reviews
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

# Remuevo el signo '$'
df['Price'] = df['Price'].apply(lambda x: x.replace('$', "") if '$' in str(x) else x)
df['Price'] = df['Price'].apply(lambda x: x.replace('Everyone', "") if 'Everyone' in str(x) else x)

df.dropna(inplace=True)

#conversion de el fomato de Price de object a float
df['Price'] = df['Price'].astype(float)

df.info()

x = df[['Reviews','Price']]
y= df['Rating']

regressor = RandomForestRegressor()
regressor.fit(x, y)

joblib.dump(regressor, 'regressor.pkl')


