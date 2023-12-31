import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
from streamlit_option_menu import option_menu
import pydeck as pdk



st.set_page_config(
    page_title="Urban_1",
    page_icon=":world_map:️",
    layout="wide",
)

selecter = option_menu(None, ["Infos", "Classification", "Segmentation",], icons=['bi-info-circle-fill', 'bi-diagram-3-fill', "bi-houses-fill"], 
                        menu_icon="cast", default_index=0, orientation="horizontal")

#----------------------------------------------------------------
@st.cache_data(show_spinner="Loading the data")
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
if selecter == "Infos":
    from streamlit_player import st_player

    # Embed a youtube video
    st_player("https://youtu.be/SQTQaE7-cpY")

    # st.image( "https://www.sinesolecinema.com/wp-content/uploads/2018/11/cropped-work-in-progress.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    st.stop()



    


#----------------------------------------------------------------
SAMPLER = st.sidebar.slider(label="Chose the sample", min_value=0.5, max_value=1.0, value=0.8, step=0.01)

left_1,right_1 = st.columns(spec=2, gap="medium")

with left_1:
    with st.expander("With outlines"):
        tab_1a,tab_1b = st.tabs([":bar_chart:", ":bookmark_tabs:"])
        
        fig = sns.pairplot(df_model[['Price', 'Area', 'Room']], diag_kind='auto',corner=True)
    
        tab_1a.dataframe(df_model.describe())
        tab_1b.pyplot(fig)


with right_1:
    with st.expander("Without outlines"):
        tab_2a,tab_2b = st.tabs([":bar_chart:", ":bookmark_tabs:"])
        
        area  = df_model["Area"].quantile(SAMPLER)
        price  = df_model["Price"].quantile(SAMPLER)
        room  = df_model["Room"].quantile(SAMPLER)
        df_model_class = df_model[(df_model["Area"]<=area)&(df_model["Price"]<=price)&(df_model["Room"]<=room)]
        st.sidebar.markdown(f"The size of the new sample is {len(df_model_class)} on {len(df_model)}")
        st.sidebar.divider()
        
        fig_2 = sns.pairplot(df_model_class[['Price', 'Area', 'Room']], diag_kind='auto',corner=True)
        
        tab_2a.dataframe(df_model_class.describe())
        tab_2b.pyplot(fig_2)

        

st.divider()

if selecter == "Classification":
    
    from sklearn import set_config
    from sklearn.utils import resample
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder

    from sklearn.metrics import precision_recall_fscore_support as score

    col1,col2 = st.columns([2,4])

    CLASS_PRICE = col1.slider(label="Select the price", min_value=int(df_model_class.describe().loc["25%","Price"]), 
                                    max_value=int(df_model_class.describe().loc["75%","Price"]), value=int(df_model_class.describe().loc["50%","Price"]), 
                                    step=1000)

    st.divider() 
    
    df_model_class['price_class'] = pd.cut(df_model_class.Price,
                                             bins=[df_model_class["Price"].min(),
                                                   CLASS_PRICE,
                                                   df_model_class["Price"].max()],
                                             include_lowest=True,
                                             labels=['low','high']
                                              )

    
    # @st.cache_resource(experimental_allow_widgets=True)
    # def model():        
        
    low = df_model_class[df_model_class.price_class == 'low']
    high = df_model_class[df_model_class.price_class == 'high']
    high_oversampled = resample(high, replace=True, n_samples=len(low))
    oversampled = pd.concat([low, high_oversampled])
    
    df_model_2 = oversampled.iloc[:,1:]
    
    X = df_model_2.iloc[:,:-1]

    le = LabelEncoder()
    y = le.fit_transform(df_model_2.price_class)
    
    categorical_columns = df_model_2.select_dtypes('object').columns.tolist()
    numerical_columns = df_model_2.select_dtypes('int64').columns.tolist()
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    
    # encode the categories
    categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    
    # fill NA values with mean and standardize.
    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
         ('standardizer', StandardScaler())
    ])
    
    # preprocessing
    preprocessing = ColumnTransformer(
        [('cat', categorical_encoder, categorical_columns),
         ('num', numerical_pipe, numerical_columns)])
    
    
    dict_model = {"Ada Boost Classifier":AdaBoostClassifier(),
                "Extra Trees Classifier":ExtraTreesClassifier(),
                "Gradient Boosting Classifier":GradientBoostingClassifier(),
                "Random Forest Classifier":RandomForestClassifier(),
                 }
    
   
    
    
    MODEL = st.sidebar.selectbox(label="Chose a model", options=list(dict_model), disabled=False, label_visibility="visible")
    st.sidebar.divider()
    # create the pipeline
    rf = Pipeline([
        ('preprocess', preprocessing),
        ('classifier', dict_model[MODEL])
    ])
    
    # fit the pipeline
    rf.fit(X_train, y_train)
        

    y_true = le.inverse_transform(y_test)
    y_pred = le.inverse_transform(rf.predict(X_test))
    
    precision, recall, fscore, support = score(y_true, y_pred)
           
    data = {"Recall":recall,
            "Precision":precision,
            "F1 score":fscore
            }
    
    st.sidebar.markdown("Model metrics")
    st.sidebar.dataframe(pd.DataFrame(data=data,index=["High","Low"]).round(2).T)
    
    
    
    AREA = col1.slider(label="Chose area", min_value=20, max_value=150, value=30, step=1)
    ROOM = col1.slider(label="Chose rooms", min_value=1, max_value=10, value=2, step=1)
    GEBIED = col1.selectbox(label="Chose neighbour", options=df_model_class.Gebied.unique(), disabled=False, label_visibility="visible")
    
    data = {'Area':AREA, 'Room':ROOM, 'Gebied':GEBIED}
    df_predict = pd.DataFrame(data,index=range(1))

    predict = le.inverse_transform(rf.predict(df_predict))

    if predict == 'high':
        col2.image( "https://th.bing.com/th/id/R.f2698a170e643bc3ee8964a9453a0b0b?rik=9pQvOpBLAM30Pw&riu=http%3a%2f%2fwww.clipartbest.com%2fcliparts%2fRTd%2f6z9%2fRTd6z9rRc.png&ehk=4ub9BMKBWN%2fwOZIq0RP4DJdH05paYdloeZFT4hNqB0E%3d&risl=&pid=ImgRaw&r=0", caption=None, width=300, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
        
    elif predict == 'low':
        col2.image( "https://media.istockphoto.com/id/174685030/photo/recession-chart.jpg?s=170667a&w=0&k=20&c=gBkfj3hmsff0hh727UjUms8s2uuBz6bMRknqjpGgEio=", caption=None, width=300, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
       


#----------------------------------------------------------------
elif selecter == "Segmentation":


    #----------------------------------------------------------------
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
        
    
    KLUSTER = st.sidebar.number_input(label=f"Chose the number of clusters. Consider that the elbow is reached with {kl.elbow} clusters", 
                                      min_value=2, max_value=5, value=kl.elbow, step=1,  disabled=False, label_visibility="visible")
    
    kmeans = cluster.KMeans(n_clusters=KLUSTER ,**kmeans_kwargs)
    kmeans = kmeans.fit(X)
    
    df_segmentation['Clusters'] = kmeans.labels_ + 1 
    
    st.sidebar.dataframe(df_segmentation['Clusters'].value_counts().to_frame())
    
    #---------------------------

    

    tab_3a,tab_3b,tab_3c,tab_3d = st.tabs(["📏 size", "💶 price","🚪 rooms","📌 maps"])
    #------------------
    import altair as alt
    
    source = df_segmentation
    
    base = alt.Chart(source)
    
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
    
    source_2 = df_segmentation.groupby(["Clusters","Room"],as_index=False).size()

    Room = alt.Chart(source_2).mark_bar().encode(
        y=alt.Y('sum(size)', stack="normalize"),
        x='Clusters:N',
        color='Room:N'
    )
    
    
    
    
    #--------------------
    tab_3a.altair_chart(altair_chart=Area, use_container_width=True, theme="streamlit")
    tab_3b.altair_chart(altair_chart=Price, use_container_width=True, theme="streamlit")
    tab_3c.altair_chart(altair_chart=Room, use_container_width=True, theme="streamlit")


    #---------------------
    with tab_3d:
        
        
        df = gpd.GeoDataFrame(pd.merge(df_segmentation, 
                                       gdf_areas_point[['Address', 'Zip','geometry']],
                                       left_index=True, 
                                       right_index=True),
                              geometry='geometry',
                              crs="EPSG:4326")
        
    
        colors = dict(zip(df.Clusters.sort_values().unique().tolist(),
                  list(sns.color_palette("husl", len(df.Clusters.sort_values().unique())))
                 )
             )
    
        df['color'] = df["Clusters"].map(colors).apply(lambda x: [i*255 for i in x])
        df['City'] = df['Address'].str.split(",",n=1,expand=True)[1]
        df['Address'] = df['Address'].str.split(",",n=1,expand=True)[0]

        
        CLUSTERS = st.multiselect('Chose which cluster',df["Clusters"].unique(),df["Clusters"].unique())
        df = df[df["Clusters"].isin(CLUSTERS)]
        

        MAP = st.radio(label="Chose a layer style", options=["Screen Grid Layer","Grid Layer","Scatter plot Layer"],horizontal=True)
        
        if MAP == "Screen Grid Layer":
            # Define a layer to display on a map
            layer = pdk.Layer(
                "ScreenGridLayer",
                df,
                pickable=True,
                opacity=0.8,
                cell_size_pixels=50,
                color_range=[
                    [0, 25, 0, 25],
                    [0, 85, 0, 85],
                    [0, 127, 0, 127],
                    [0, 170, 0, 170],
                    [0, 190, 0, 190],
                    [0, 255, 0, 255],
                ],
                get_position="geometry.coordinates",
            )
            
            # Set the viewport location
            view_state = pdk.ViewState(latitude=52.370978, longitude=4.899875, zoom=12, bearing=0, pitch=0)
            
            tooltip = {
                "html": "<b>{cellCount} properties",
                "style": {"background": "#DC851F", "color": "#45462a", "font-family": '"Helvetica Neue", Arial'},
            }
            
        
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)

            st.pydeck_chart(pydeck_obj=r, use_container_width=True)

        elif MAP == "Grid Layer":
            layer = pdk.Layer(
                "GridLayer", df, pickable=True, extruded=True, cell_size=200, elevation_scale=4, get_position="geometry.coordinates",
            )

            # Set the viewport location
            view_state = pdk.ViewState(latitude=52.370978, longitude=4.899875, zoom=12, bearing=0, pitch=45)
            
            tooltip = {
                "html": "<b>{count} properties",
                "style": {"background": "#DC851F", "color": "#45462a", "font-family": '"Helvetica Neue", Arial'},
            }
            
        
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)

            st.pydeck_chart(pydeck_obj=r, use_container_width=True)

        elif MAP == "Scatter plot Layer":
            left_2,right_2 = st.columns(spec=[1,1], gap="medium")
        
            
            RATIO_SCALE = left_2.number_input(label=f"Ratio scale", min_value=0.1, max_value=30.0, value=1.0, step=0.1)
            GET_RATIO = right_2.selectbox(label="Select a variable", options=['Price', 'Area', 'Room'], disabled=False, label_visibility="visible")
        
            # ratio_scale = 
        
            if GET_RATIO == 'Price':
                GET_RATIO = "Price/1000"
        
            # Define a layer to display on a map
            layer = pdk.Layer(
                "ScatterplotLayer",
                df,
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                radius_scale=RATIO_SCALE,
                line_width_min_pixels=1,
                get_position="geometry.coordinates",
                get_radius=GET_RATIO,
                get_fill_color='color',
                get_line_color=[0, 0, 0],
            )
            
            # Set the viewport location
            view_state = pdk.ViewState(latitude=52.370978, longitude=4.899875, zoom=12, bearing=0, pitch=0)
            
            TOOLTIP = {'Price/1000':{"text": "Price: {Price} euros \n{Address}, {Zip} {City}"}, 
                       'Area':{"text": "Dimension: {Area} squared meters \n{Address}, {Zip} {City}"},
                       'Room':{"text": "Number of rooms: {Room} \n{Address}, {Zip} {City}"}}
            
            
            r = pdk.Deck(layers=[layer], 
                 initial_view_state=view_state,
                tooltip=TOOLTIP[GET_RATIO])
    
            st.pydeck_chart(pydeck_obj=r, use_container_width=True)
