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


# #----------------------------------------------------------------
# df_3 = df_model_class.groupby(['price_class']).mean()#.round(2)
# st.dataframe(df_3)


# #----------------------------------------------------------------
# fig_3 = df_model_class.price_class.value_counts().plot(kind='bar')
# st.pyplot(fig_3)


# #----------------------------------------------------------------
# df_4 = df_model_class.groupby('Gebied').mean().round() \
# .sort_values('Price', ascending=False)
# st.dataframe(df_4)


# #----------------------------------------------------------------
# fig_4 = df_model_class.Gebied.value_counts().plot(kind='bar')
# st.pyplot(fig_4)


#----------------------------------------------------------------
from sklearn import set_config
from sklearn.utils import resample
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingRegressor, ExtraTreesClassifier, GradientBoostingClassifier#, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder


low = df_model_class[df_model_class.price_class == 'low']
high = df_model_class[df_model_class.price_class == 'high']
high_oversampled = resample(high, replace=True, n_samples=len(low))
low_subsampled = resample(low, replace=False, n_samples=len(high))
oversampled = pd.concat([low, high_oversampled])
subsampled = pd.concat([high, low_subsampled])

df_model_2 = oversampled.iloc[:,1:]

X = df_model_2.iloc[:,:-1]
le = LabelEncoder()
y = le.fit_transform(df_model_2.price_class)

categorical_columns = df_model_2.select_dtypes('object').columns.tolist()
numerical_columns = df_model_2.select_dtypes('int64').columns.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y,)

# encode the categories
categorical_encoder = OneHotEncoder(handle_unknown='ignore')

# fill NA values with mean and standardize.
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
     ('standardizer' , StandardScaler())
])

# preprocessing
preprocessing = ColumnTransformer(
    [('cat', categorical_encoder, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])


dict_model = {"Ada Boost Classifier":AdaBoostClassifier(),
            "Bagging Regressor":BaggingRegressor(),
            "Extra Trees Classifier":ExtraTreesClassifier(),
            "Gradient Boosting Classifier":GradientBoostingClassifier(),
            "Random Forest Classifier":RandomForestClassifier(),
            # "Voting Classifier":VotingClassifier()
             }

MODEL = st.selectbox(label="Chose a model", options=list(dict_model), disabled=False, label_visibility="visible")


# create the pipeline
rf = Pipeline([
    ('preprocess', preprocessing),
    ('classifier', dict_model[MODEL])
])

# fit the pipeline
rf.fit(X_train, y_train)

st.write(f"{MODEL} train accuracy: %0.3f" % rf.score(X_train, y_train))
st.write(f"{MODEL} test accuracy: %0.3f" % rf.score(X_test, y_test))



y_true = le.inverse_transform(y_test)
y_pred = le.inverse_transform(rf.predict(X_test))
confusion_matrix = pd.crosstab(le.inverse_transform(y_test),
                               le.inverse_transform(rf.predict(X_test)),
                               rownames=['Actual'], colnames=['Predicted'],
                               normalize='index')

st.dataframe(confusion_matrix)


import sklearn.cluster as cluster
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler

# Standardizing the features
df_segmentation = df_model_class.iloc[:,:3]
X = MinMaxScaler().fit_transform(df_segmentation)

kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = cluster.KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    
kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)

st.write(f'The elbow is reached with {kl.elbow} clusters')


KLUSTER = st.number_input(label="Chose the number of clusters", min_value=2, max_value=5, value=kl.elbow, step=1,  disabled=False, label_visibility="visible")

kmeans = cluster.KMeans(n_clusters=KLUSTER ,**kmeans_kwargs)
kmeans = kmeans.fit(X)

df_segmentation['Clusters'] = kmeans.labels_ + 1 

st.dataframe(df_segmentation['Clusters'].value_counts().to_frame())


#------------------
import altair as alt

source = df_segmentation

base = alt.Chart(source).properties(width=80,height=300)

Price = base.mark_boxplot().encode(
    x=alt.X('Clusters:N',title=None,axis=alt.Axis(values=[0], ticks=True, grid=False, labels=True)),
    y=alt.Y('Price:Q', type='quantitative',axis=alt.Axis(ticks=True, grid=False, labels=True)),
    color='Clusters:N'
)

Area = base.mark_boxplot().encode(
    x=alt.X('Clusters:N',title=None,axis=alt.Axis(values=[0], ticks=True, grid=False, labels=True)),
    y=alt.Y('Area:Q', type='quantitative',axis=alt.Axis(ticks=True, grid=False, labels=True)),
    color='Clusters:N'
)


Room = base.mark_bar().encode(
    column='Clusters:N',
    x=alt.X('Room:O'),
    y=alt.Y('count()'),
    color='Clusters:N'
)

#--------------------
# cm = sns.light_palette("green", as_cmap=True)
# cluster_mean = df_segmentation.groupby('Clusters')[['Price','Area']].mean()
# st.dataframe(cluster_mean.style.background_gradient(cmap=cm).set_precision(1))


#--------------------
st.altair_chart(altair_chart=alt.hconcat(Price,Area), use_container_width=False, theme="streamlit")









