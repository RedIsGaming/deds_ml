# # Week 8 Supervised Learning

# ## Practicum 8-1
# 
# PR8-1: Regressiemodellen in Machine Learning. Great Outdoors wil graag weten hoeveel zij gaat verkopen op basis van een set onafhankelijke variabelen. Daarom wil zij een model trainen op basis van reeds bekende data, zodat deze volgend jaar in gebruik kan worden genomen. Je doet dus het volgende met de reeds bekende data:
# 
# * Bedenk met welke onafhankelijke variabelen, die ook uit meerdere databasetabellen kunnen komen, dit naar verwachting het beste voorspeld kan worden en zet deze samen met de afhankelijke variabele in één DataFrame.
# * Pas waar nodig Dummy Encoding toe.
# * Snijd dit DataFrame horizontaal en verticaal op de juiste manier.
# * Train het regressiemodel.
# * Evalueer de performance van je getrainde regressiemodel.

# ### Tabels 1/2
# 
# Hieronder zie je de tabellen die we gaan gebruiken voor deze practicumopdracht van 8-1.

# age_group
# sales_demographic
# sales_territory

# order_details
# order_header
# order_method

# product
# product_line
# product_type

# sales_branch
# SALES_TARGETData

# ### Imports
# 
# Hier staan alle imports die gebruikt zullen worden voor de practicumopdrachten 8-1 en 8-2.

import sqlite3 as sql
import warnings as warn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, metrics

from typing import Any
warn.filterwarnings("ignore")

# ### SQLite connections and CSV-files
# 
# Hier maken we de connectie naar de SQLite Databases en roepen we de CSV bestanden aan.

conn1 = sql.connect("Great_Outdoors_Data_SQLite/go_crm.sqlite")
conn2 = sql.connect("Great_Outdoors_Data_SQLite/go_sales.sqlite")
conn3 = sql.connect("Great_Outdoors_Data_SQLite/go_staff.sqlite")

go_sales_inventory_levelsdata = pd.read_csv("Great_Outdoors_Data_SQLite/GO_SALES_INVENTORY_LEVELSData.csv")
go_sales_product_forecastdata = pd.read_csv("Great_Outdoors_Data_SQLite/GO_SALES_PRODUCT_FORECASTData.csv")

# ### Check all tables
# 
# Hieronder wordt gecheckt welke tabellen er nou in Python zitten.

print(globals().keys())

# ### Check all tables of the SQLite databases.
# 
# Hier checken we alle DataFrames van de tabellen voor een snel overzicht.

# Iterate over all tables from the conns.
for i, conn in enumerate([conn1, conn2, conn3]):
    tables: list[Any] = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

    for table in tables:
        globals()[table[0] + str(i)] = pd.read_sql_query(f"SELECT * FROM {table[0]}", conn)
        print(globals()[table[0] + str(i)])

# ### For quick testing the necassery tables
# 
# We kunnen snel de inhoud zien van de tabellen.

test = pd.read_sql_query(f"SELECT * FROM SALES_TARGETData", conn2)
test

# TRIAL column not taken into account
# age_group => 3 columns
# sales_demographic => 4 columns
# sales_territory => 2 columns
# 9 columns

# order_details => 7 columns
# order_header => 8 columns
# order_method => 2 columns
# 17 columns

# product => 9 columns
# product_line => 2 columns
# product_type => 3 columns
# 14 columns

# sales_branch => 7 columns
# SALES_TARGETData => 8 columns
# 15 columns
# 55 columns in total - 10 columns = 45 columns

# ### Tables 2/2
# 
# Tabellen die we gaan gebruiken voor het trainen en testen van de data voor opdracht 8-1.

# Remove tables that we not gonna use.
go_tables: list[str] = ["age_group0", "sales_demographic0", "sales_territory0", "order_details1", "order_header1", 
    "order_method1", "product1", "product_line1", "product_type1", "sales_branch1", "SALES_TARGETData1"
]

# ### Merge the tables
# 
# Hier voegen we de tabellen samen, zodat we betere voorspellingen kunnen maken.

def go_merge(go_table: Any):
    df_merge: list[Any] = [globals()[table] for table in go_table]
    merged_df = pd.concat(df_merge, ignore_index=True)

    return merged_df

# ### Drop unnecassary columns 1/2
# 
# Hier verwijderen we de nutteloze TRIAL kolommen.

trial = go_merge(go_tables)
df = trial[trial.columns.drop(list(trial.filter(regex="TRIAL")))]

# ### Show all the columns
# 
# We tonen alle kolommen en filteren de lege waardes eruit bij de volgende stap.

#Here we display all the columns
pd.set_option("display.max_columns", None)
df

# ### Delete unnecassary columns 2/2
# 
# Hier verwijderen we kolommen die geen waarde hebben op het regressiemodel.

df = df.drop(columns=[
    "RETAILER_NAME", "ADDRESS1", "ADDRESS2", "CITY", "REGION", "POSTAL_ZONE", "PRODUCT_IMAGE", "PRODUCT_NAME", "DESCRIPTION",
    "TERRITORY_NAME_EN", "ORDER_METHOD_EN", "LANGUAGE", "PRODUCT_LINE_EN", "PRODUCT_TYPE_EN"
])
df = df.fillna(0)
df

# ### Convert the columns to numeric types
# 
# Alle kolommen worden naar numerieke waardes omgezet, zodat ze bruikbaar zijn voor de regressiemodellen.

# Convert all remaining columns to numeric types
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# ### Show the predictions
# 
# Met Matplotlib kunnen we een grafiek maken en voorspellingen doen.

colors = np.interp(df["SALES_PERCENT"], (df["SALES_PERCENT"].min(), df["SALES_PERCENT"].max()), (0, 66))

plt.xlabel("Demographic Code")
plt.ylabel("Sales Percent")
plt.title("Sales")
plt.scatter(x=df["DEMOGRAPHIC_CODE"], y=df["SALES_PERCENT"], s=25, c=colors, cmap="GnBu", alpha=0.8)
plt.colorbar(orientation="vertical", label="Percentage Sale", extend="both")
plt.clim(0, 100)
plt.show()

# ### One-Hot encoding the SALES_YEAR column
# 
# Hier gaan we een dummy gebruiken voor het encoden van een niet numerieke variabele.

df_dummies = pd.get_dummies(df.loc[:, ["SALES_YEAR"]])
df_dummies

# ### Add the dummy column to the DataFrame and remove the old column
# 
# We zetten de dummy kolom in het DataFrame en verwijderen de oude kolom, zodat we dit kunnen gebruiken voor regressiemodel.

df = pd.concat([df, df_dummies], axis=1)
df = df.drop(columns=["SALES_YEAR"], axis=1)
df

# ### Horizontal and vertical cut
# 
# We snijden horizontaal en verticaal, zodat we het model kunnen trainen en testen. Ook vullen we defaults in voor X.

X = df.drop(columns=["SALES_PERCENT"])
y = df["SALES_PERCENT"]
X = X.fillna(0)
X

# ### Train and test data
# 
# Hier kunnen we eindelijk de data trainen en testen die we nodig hebben.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model: LinearRegression = LinearRegression()
model.fit(X_train, y_train)
model

# ### Predict the model
# 
# We kunnen het model met de test data van de onafhankelijke waarde voorspellen.

y_pred = model.predict(X_test)
y_pred

# ### Get a new predicted column
# 
# We maken een nieuwe kolom aan met een nieuwe naam die we met het oude kolom kunnen vergelijken.

df_pred = pd.DataFrame(y_pred)
df_pred = df_pred.rename(columns={0: "PREDICTED_SALES_PERCENT"})
df_pred

# ### Concat old column with new column
# 
# We vergelijken beide kolommen door ze samen te voegen.

y_test_merge = pd.concat([y_test.reset_index()["SALES_PERCENT"], df_pred], axis=1)
y_test_merge.loc[y_test_merge["PREDICTED_SALES_PERCENT"].notna(), :]

# ### Show the predictions for the columns
# 
# Met Matplotlib kunnen we wederom weer in een grafiek de tabellen met elkaar vergelijken en voorspellen.

colors = np.interp(y_test_merge["SALES_PERCENT"], (y_test_merge["SALES_PERCENT"].min(), y_test_merge["SALES_PERCENT"].max()), (0, 66))

plt.xlabel("Predicted Sales Percent")
plt.ylabel("Sales Percent")
plt.title("Sales")
plt.scatter(x=y_test_merge["PREDICTED_SALES_PERCENT"], y=y_test_merge["SALES_PERCENT"], s=25, c=colors, cmap="GnBu", alpha=0.8)
plt.colorbar(orientation="vertical", label="Percentage Sale", extend="both")
plt.clim(0, 100)
plt.show()

# ### Measure mean errors for squared and absolute
# 
# We berekenen voor beide squared en absolute de error, zodat we kunnen zien hoe goed het model werkt.

mean_squared_error(y_test_merge["SALES_PERCENT"], y_test_merge["PREDICTED_SALES_PERCENT"])

mean_absolute_error(y_test_merge["SALES_PERCENT"], y_test_merge["PREDICTED_SALES_PERCENT"])

# ## Practicum 8-2
# 
# PR8-2: Classificatiemodellen in Machine Learning. Great Outdoors wil graag weten wat de retourredenen gaan zijn op basis van een set onafhankelijke variabelen. Daarom wil zij een model trainen op basis van reeds bekende data, zodat deze volgend jaar in gebruik kan worden genomen. Let op: de retourreden kan ook "n.v.t." zijn, niet elke order wordt namelijk geretourneerd; je zult dit moeten aanpakken door een join tussen "returned_item" en "order_details". Je doet dus het volgende met de reeds bekende data:
# 
# * Bedenk met welke onafhankelijke variabelen dit naar verwachting het beste voorspeld kan worden en zet deze samen met de afhankelijke variabele in één DataFrame.
# * Pas waar nodig Dummy Encoding toe.
# * Snijd dit DataFrame horizontaal en verticaal op de juiste manier.
# * Train het classificatiemodel.
# * Evalueer de performance van je getrainde classificatiemodel a.d.h.v. een confusion matrix.
# 

# ### Tables to use for the classification model
# 
# De tabellen die we gebruiken voor het classificatie model die we ook snel kunnen uitlezen.

# return_reason
# returned_item
# order_details

return_reason = pd.read_sql_query(f"SELECT * FROM return_reason", conn2)
returned_item = pd.read_sql_query(f"SELECT * FROM returned_item", conn2)
order_details = pd.read_sql_query(f"SELECT * FROM order_details", conn2)
returned_item

# TRIAL column not taken into account
# return_reason => 2 columns
# returned_item => 5 columns
# order_details => 7 columns
# 14 columns

# ### Merging the used tables together
# 
# We zullen hier alle tabellen samenvoegen om dit te kunnen gebruiken voor het model en we vegen de TRIAL kolommen schoon.

df = returned_item.merge(order_details, how="outer", on="ORDER_DETAIL_CODE").merge(return_reason, how="outer", on="RETURN_REASON_CODE", indicator=True)
df = df[df.columns.drop(list(df.filter(regex="TRIAL")))]
df = df.dropna()
df

# ### Convert the return date to a string
# 
# We moeten de retourdatum converteren naar een string, want anders werkt het plotten niet.

df["RETURN_DATE"].to_string()
df

# ### Create dummies for the return description
# 
# We zullen dummies maken van de retourbeschrijving, zodat we meer gegevens hebben.

df_dummies = pd.get_dummies(df.loc[:, ["RETURN_DESCRIPTION_EN"]])
df_dummies

# ### Add the dummy columns to the DataFrame
# 
# Hier voegen we de dummy kolommen toe aan het DataFrame en verwijderen we de oude.

df = pd.concat([df, df_dummies], axis=1)
df = df.drop(columns=["RETURN_DESCRIPTION_EN"], axis=1)
df

# ### Drop the return description and put it to the test data and horizontal/vertical cut
# 
# We zullen de retourredenen verwijderen en de dummy data gebruiken voor het testen. Hier wordt horizontaal en verticaal gesneden.

X = df.drop(columns=["RETURN_DESCRIPTION_EN_Defective product", "RETURN_DESCRIPTION_EN_Incomplete product", "RETURN_DESCRIPTION_EN_Unsatisfactory product", "RETURN_DESCRIPTION_EN_Wrong product ordered", "RETURN_DESCRIPTION_EN_Wrong product shipped"])
y = df[["RETURN_DESCRIPTION_EN_Defective product", "RETURN_DESCRIPTION_EN_Incomplete product", "RETURN_DESCRIPTION_EN_Unsatisfactory product", "RETURN_DESCRIPTION_EN_Wrong product ordered", "RETURN_DESCRIPTION_EN_Wrong product shipped"]]
y

# ### Train and test the model
# 
# Hieronder zullen we voor de juiste data het model trainen en testen.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
list(X.columns)

# ### A tree with a max depth
# 
# Hieronder is een grafiek waarin je de maximale diepte kan inzien.

model: RandomForestClassifier = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(X_train, y_train)
tree.plot_tree(model, feature_names=X.columns, filled=True)
plt.show()

# ### Test data to DataFrame and renamed
# 
# Hier zullen we de onafhankelijk test data naar een DataFrame omzetten en de kolom hernoemen.

df_pred = pd.DataFrame(model.predict(X_test))
df_pred = df_pred.rename(columns={0: "PREDICTED_RETURN_DESCRIPTION_EN"})
frame = pd.concat([y_test.reset_index()["RETURN_DESCRIPTION_EN"], df_pred], axis=1)
frame

# ### Confusion Matrix
# 
# We hebben hier een afbeelding met een Confusion Matrix, zodat er snel voorspellingen kunnen worden gemaakt

matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
display: metrics.ConfusionMatrixDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[False, True])
display.plot(cmap="GnBu")
plt.show()

# ### Test the accuracy score
# 
# We kunnen nu meten hoe goed het model het heeft gedaan.

metrics.accuracy_score(frame["RETURN_DESCRIPTION_EN"], frame["PREDICTED_RETURN_DESCRIPTION_EN"])

# ### Lorem
# 
# ipsum

# y_pred = model.predict(X_test)
# print(accuracy_score(y_test, y_pred))
