# %%
# 0    age_group
# 2    retailer
# 5    retailer_segment
# 6    retailer_site
# 7    retailer_type
# 8    sales_demographic
# 9    sales_territory

# 1    order_details
# 2    order_header
# 3    order_method
# 4    product
# 5    product_line
# 6    product_type
# 7    retailer_site
# 10    sales_branch
# 11    sales_staff
# 12    SALES_TARGETData

# 1    sales_branch

# %%
import sqlite3 as sql
import warnings as warn

import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import LabelEncoder

# %%
warn.filterwarnings("ignore")

# %%
conn1 = sql.connect("Great_Outdoors_Data_SQLite/go_crm.sqlite")
conn2 = sql.connect("Great_Outdoors_Data_SQLite/go_sales.sqlite")
conn3 = sql.connect("Great_Outdoors_Data_SQLite/go_staff.sqlite")

# Iterate over all tables from conn1
for i, conn in enumerate([conn1, conn2, conn3]):
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

    for table in tables:
        globals()[table[0] + str(i)] = pd.read_sql_query(f"SELECT * FROM {table[0]}", conn)

# %%
sales_product_forecast = pd.read_csv("Great_Outdoors_Data_SQLite/GO_SALES_PRODUCT_FORECASTData.csv")
sales_inventory_levels = pd.read_csv("Great_Outdoors_Data_SQLite/GO_SALES_INVENTORY_LEVELSData.csv")

# %%
# Remove tables that we not gonna use

# %%

# Laten we voor nu er vanuit gaan dat sales_percent de waarde is die we moeten voorspellen
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 0    age_group
# 8    sales_demographic
# 9    sales_territory

# 1    order_details
# 2    order_header
# 3    order_method
# 4    product
# 5    product_line
# 6    product_type
# 10    sales_branch
# 12    SALES_TARGETData

# tables = ['age_group0', 'retailer0', 'retailer_segment0', 'retailer_site0', 'retailer_type0', 'sales_demographic0', 
#           'sales_territory0', 'order_details1', 'order_header1', 'order_method1', 'product1', 'product_line1', 'product_type1', 
#           'sales_branch1', 'sales_staff1', 'SALES_TARGETData1']

tables = ['age_group0', 'sales_demographic0', 
          'sales_territory0', 'order_details1', 'order_header1', 'order_method1', 'product1', 'product_line1', 'product_type1', 
          'sales_branch1', 'SALES_TARGETData1']


def merge_dataframes(tables):
    merged_df = globals()[tables[0]]
    tables = tables[1:]

    while tables:
        for i, table in enumerate(tables):
            common_columns = set(merged_df.columns).intersection(set(globals()[table].columns))
            if common_columns:
                merged_df = pd.merge(merged_df, globals()[table], how='outer', on=list(common_columns))
                tables.pop(i)
                break
        else:
            break

    return merged_df

hi = merge_dataframes(tables)

hi = hi[hi.columns.drop(list(hi.filter(regex='TRIAL')))]


# Convert all remaining columns to numeric types
for column in hi.columns:
    hi[column] = pd.to_numeric(hi[column], errors='coerce')

hi = hi.drop(columns=["TERRITORY_NAME_EN"])

X = hi.drop(columns=['SALES_PERCENT'])
y = hi['SALES_PERCENT']

print(len(hi))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(mean_absolute_error(y_test, y_pred))



