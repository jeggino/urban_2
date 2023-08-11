import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns




st.set_page_config(
    page_title="Urban_1",
    page_icon=":world_map:️",
    layout="wide",
)

#----------------------------------------------------------------
@st.cache_data
def get_data():
  url = "https://maps.amsterdam.nl/open_geodata/geojson_lnglat.php?KAARTLAAG=GEBIEDEN22&THEMA=gebiedsindeling"
  gdf_districts = gpd.read_file(url)
  
  # import the raw data
  df_raw = pd.read_csv('HousingPrices-Amsterdam-August-2021.csv').iloc[:,1:]
  
  # create a dataset that conteins only the points located inside the polygons
  
  
  #zip the coordinates into a point object and convert to a GeoData Frame
  geometry = [Point(xy) for xy in zip(df_raw.Lon, df_raw.Lat)]
  geo_df = gpd.GeoDataFrame(df_raw, geometry=geometry,crs="EPSG:4326")
  
  # create a dataset
  gdf_areas_point = gpd.sjoin(geo_df,gdf_districts,  how='inner',op= 'intersects')\
  [['Address', 'Zip', 'Price', 'Area', 'Room', 'Lon', 'Lat', 'geometry','Gebied']]
  
  
  # Build the classification model
  # create the dataset
  df_model = gdf_areas_point[['Price', 'Area', 'Room','Gebied']]

  return gdf_areas_point, df_model

df_model = get_data()[1]
gdf_areas_point = get_data()[0]
#----------------------------------------------------------------
st.dataframe(df_model)
st.dataframe(df_model.describe())


#----------------------------------------------------------------

fig = sns.pairplot(df_model[['Price', 'Area', 'Room']], diag_kind='auto',corner=True)
st.pyplot(fig)


#----------------------------------------------------------------
area  = df_model["Area"].quantile(0.8)
price  = df_model["Price"].quantile(0.8)
room  = df_model["Room"].quantile(0.8)

df_model_class = df_model[(df_model["Area"]<=area)&(df_model["Price"]<=price)&(df_model["Room"]<=room)]
st.dataframe(df_model_class.describe())


#----------------------------------------------------------------
fig_2 = sns.pairplot(df_model_class[['Price', 'Area', 'Room']], diag_kind='auto',corner=True)
sns.set_theme(style="white")
st.pyplot(fig_2)


#----------------------------------------------------------------
df_model_class['price_class'] = pd.cut(df_model_class.Price,
                                 bins=[df_model_class["Price"].min(),
                                       df_model_class["Price"].mean(),
                                       df_model_class["Price"].max()],
                                 include_lowest=True,
                                 labels=['low','high'])


st.dataframe(df_model_class)


#----------------------------------------------------------------
df_3 = df_model_class.groupby(['price_class']).mean()#.round(2)
st.dataframe(df_3)


#----------------------------------------------------------------
fig_3 = df_model_class.price_class.value_counts().plot(kind='bar')
st.pyplot(fig_3)


#----------------------------------------------------------------
df_4 = df_model_class.groupby('Gebied').mean().round() \
.sort_values('Price', ascending=False)
st.dataframe(df_4)


#----------------------------------------------------------------
fig_4 = df_model_class.Gebied.value_counts().plot(kind='bar')
st.pyplot(fig_4)


#----------------------------------------------------------------






