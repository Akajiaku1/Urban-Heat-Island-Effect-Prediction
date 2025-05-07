 # Predicting Urban Heat Island (UHI) Effects#

Urban Heat Island (UHI) analysis combines satellite, climate, and urban data to model city-scale temperature anomalies. We assemble open data sources, geospatial Python libraries, and machine learning to predict UHI intensity. The workflow below outlines data acquisition, preprocessing (feature extraction, cleaning, normalization), model training, and visualization. Code snippets illustrate key steps, and open-source tools (GeoPandas, Rasterio, xarray, scikit-learn, etc.) ensure reproducibility.
1. Data Sources

Gather diverse datasets covering surface temperature, weather, and urban land cover. For example:

    Satellite Imagery (Landsat/MODIS): Use thermal bands to derive Land Surface Temperature (LST) and optical bands for vegetation/build-up indices. (e.g. Landsat 8’s Band 10 for LST, Bands 4/5 for NDVI)
    landsat.visibleearth.nasa.gov
    earthpy.readthedocs.io
    . NASA provides MODIS LST products (e.g. MOD11A2) and Landsat archives (USGS EarthExplorer). Code example (EarthPy) shows NDVI calculation from Landsat 8 Red/NIR bands
    earthpy.readthedocs.io
    :

import rasterio
import numpy as np

# Load Landsat 8 red (band 4) and NIR (band 5)
red_src = rasterio.open('LC08_red.tif');  red = red_src.read(1).astype(float)
nir_src = rasterio.open('LC08_nir.tif');  nir = nir_src.read(1).astype(float)
# Compute NDVI = (NIR - Red) / (NIR + Red)
ndvi = (nir - red) / (nir + red)  # NDVI calculation:contentReference[oaicite:3]{index=3}

Meteorological Reanalysis (ERA5, ERA-Interim): Incorporate regional climate variables (air temperature, humidity, wind) using the ECMWF ERA5 dataset. The CDSAPI Python client can download ERA5 (e.g., 2m air temperature, boundary layer height) for the city region
mdpi.com
. For example:

import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-era5-single-levels',
           {'product_type': 'reanalysis',
            'variable': ['2m_temperature','boundary_layer_height'],
            'year': '2023', 'month': '06', 'day': '15',
            'time': ['12:00'], 
            'area': [lat_max, lon_min, lat_min, lon_max],  # N,W,S,E
            'format': 'netcdf'},
           'era5_20230615.nc')

These variables serve as large-scale weather predictors; prior studies show that using ERA5 reanalysis improves UHI modeling
mdpi.com
.

Urban Land Use / OpenStreetMap (OSM): Extract built-up and land-cover features. For instance, OSMnx (a Python library) can download building footprints or land-use polygons for a city
geoffboeing.com
. This yields features like building density or green-space fraction. Example usage of OSMnx:

    import osmnx as ox
    # Download all building footprints in a city polygon
    city = "Chicago, Illinois, USA"
    tags = {'building': True}
    buildings = ox.geometries_from_place(city, tags)

    OSMnx can similarly retrieve street networks or land-use areas (roads, parks, water, etc.)
    geoffboeing.com
    . These spatial features (e.g. percentage impervious area) are important UHI predictors.

Collecting these datasets in open formats (GeoTIFF, NetCDF, Shapefiles) ensures a reproducible workflow.
2. Preprocessing and Feature Engineering

Preprocess raw data into ML-ready features, handling coordinate alignment, missing data, and index calculation:

    Spatial Alignment & Subsetting: Reproject all data to a common CRS (e.g. EPSG:4326 or a local projection). Use Rasterio or rioxarray to resample or clip rasters to the study region. For example, to mask out only the city boundaries:

import rioxarray as rxr
city_raster = rxr.open_rasterio('landsat_thermal.tif', masked=True)
# Optionally clip to city shapefile using GeoPandas
import geopandas as gpd
city_shape = gpd.read_file('city_boundary.shp')
city_raster = city_raster.rio.clip(city_shape.geometry, city_shape.crs)

Derived Indices: Compute spectral indices that correlate with UHI drivers:

    NDVI (Vegetation Index): Ratio highlighting vegetation cover
    earthpy.readthedocs.io
    . Values range [-1,1], where higher values indicate greener, cooler areas. (Code above).

    NDBI (Built-up Index): Highlights impervious surfaces
    pro.arcgis.com
    , typically using SWIR and NIR bands. Formula: NDBI = (SWIR - NIR) / (SWIR + NIR)
    pro.arcgis.com
    . E.g., Landsat 8 SWIR (Band 6) minus NIR (Band 5):

    swir_src = rasterio.open('LC08_swir.tif'); swir = swir_src.read(1).astype(float)
    ndbi = (swir - nir) / (swir + nir)  # NDBI:contentReference[oaicite:11]{index=11}

    Other Indices: E.g., NDWI (water index) or MNDWI, Albedo, Sky View Factor from LiDAR or imagery, if available.

Statistical Features: From meteorological time series (ERA5 or station data), extract features like daily max/min temperature, humidity averages, or rolling means. Compute temporal features (hour of day, day of year) to capture seasonal/diurnal cycles.

Urban Geometry Features: From OSM building footprints, calculate metrics such as building density or height (if available), per-pixel impervious fraction, or distance to nearest park.

Missing Data Handling: Many spatial datasets have gaps (e.g., cloud cover in optical imagery). Fill missing values via imputation (e.g. mean/median) or masking. In code:

import numpy as np
# Example: mask clouds (assuming masked array)
ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)  # set non-finite to nan
# Impute with mean
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
ndvi_flat = ndvi.reshape(-1, 1)
ndvi_filled = imp.fit_transform(ndvi_flat).reshape(ndvi.shape)

Normalization/Scaling: Scale features to similar ranges (important for ML). For instance, use StandardScaler or min-max scaling on all input features:

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # fit on training data

Careful collation of features (vector or raster stacks, e.g. via xarray or stacking arrays) is required. The final dataset can be structured as a table (rows = locations or pixels, columns = features + target temperature difference).
3. Machine Learning Model

With features prepared, train a regression model to predict UHI intensity (e.g., surface air or land temperature anomaly). Common choices include Random Forest, Gradient Boosting, or Neural Networks
mdpi.com
. For example:

    Random Forest Regression: Handles nonlinearities and mixed data well. Example code using scikit-learn:

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assume X (features) and y (target ΔT) are prepared NumPy arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=200, random_state=0)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)

Gradient Boosting / XGBoost: Often yields strong performance. For example:

from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)

Neural Networks: For spatial-temporal patterns, one might use a simple MLP or more advanced CNN/LSTM architectures. Example MLP with Keras:

    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    nn_preds = model.predict(X_test).flatten()

Select the model based on data size and complexity. Recent studies found that Gradient Boosting (XGBoost) and DNNs often outperform other methods for LST prediction
sciencedirect.com
mdpi.com
. Cross-validation and hyperparameter tuning (e.g. using GridSearchCV) improve robustness.

After training, evaluate model accuracy (RMSE, R²) and check residual patterns. Feature importance (e.g. rf.feature_importances_ or SHAP analysis) can reveal which inputs (e.g. NDVI, building cover) are most influential.
4. Visualization of UHI Maps

Finally, translate model predictions into spatial maps highlighting UHI-prone zones. Use geospatial plotting libraries:

    Heatmaps (Matplotlib): If predictions are on a raster grid, simply plot with a temperature colormap:

import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.imshow(predicted_temp_map, cmap='hot', origin='upper')
plt.colorbar(label='Predicted Land Temperature (°C)')
plt.title("Predicted UHI Intensity")
plt.axis('off')
plt.show()

GeoPandas Choropleths: For region-based data (e.g. districts), merge predictions into a GeoDataFrame and plot:

    import geopandas as gpd
    city_gdf = gpd.read_file('city_districts.shp')
    city_gdf['pred_temp'] = predictions  # attach model output
    city_gdf.plot(column='pred_temp', cmap='hot', legend=True)
    plt.title("Predicted Urban Heat Island (°C)")
    plt.show()

    Interactive Maps (Folium): Overlay heat layers on web maps.

Embeddeding satellite imagery can illustrate UHI visually. For example, NASA Landsat data of New York City (August 2002) shows clear UHI patterns: urban centers (yellow/orange) are much hotter than surrounding green areas
landsat.visibleearth.nasa.gov
.

Figure: Landsat-derived surface temperature in New York City (Aug 2002). Hotspots (yellow) occur in dense urban areas, while vegetated regions (purple) remain cooler
landsat.visibleearth.nasa.gov
.

These plots confirm model outputs and help urban planners identify mitigation targets (e.g. tree planting in hottest neighborhoods).
5. Tools, Reproducibility, and Case Studies

This workflow relies on open-source Python tools:

    Rasterio / rioxarray / xarray for raster data I/O and manipulation.

    GeoPandas for vector data and mapping.

    Scikit-learn, XGBoost, TensorFlow/Keras for ML models.

    OSMnx / Overpass API for extracting urban form.

    Matplotlib / Seaborn / Folium for visualization.

Each step can be scripted in a Jupyter notebook or as a Python module, with version-controlled code. NASA’s ARSET training emphasizes this satellite-to-ML pipeline, noting that thermal satellite data and land-cover information are key to UHI mapping
appliedsciences.nasa.gov
. For example, the NASA LaSTMoV project provides Python tools for processing MODIS LST and generating heat maps
software.nasa.gov
. Additionally, open datasets like the Kaggle UHI Monitoring Dataset offer real-world examples (though one must register to access Kaggle data).

Case Study Example: In a Moscow study, Varentsov et al. used 21 years of weather station and ERA5 data to train Random Forest and Gradient Boosting models for city–rural temperature differences
mdpi.com
. They achieved ~0.7 K RMSE, demonstrating that ML can match complex climate models for UHI trends. Similarly, Gaffin et al. (2006) used Landsat thermal images to map New York’s UHI and found vegetation cover inversely correlated with LST
landsat.visibleearth.nasa.gov
.

By combining these techniques and referencing published methods, one builds a robust, reproducible Python codebase for UHI prediction. Each step—from fetching satellite/ERA5 data, through NDVI/NDBI computation
earthpy.readthedocs.io
pro.arcgis.com
, to model training and heatmap plotting—can be documented and automated, facilitating open research on urban climate.

Sources: Authoritative references and tools cited above detail the algorithms and libraries (e.g. EarthPy for NDVI
earthpy.readthedocs.io
, OSMnx for OSM data
geoffboeing.com
, and MDPI UHI studies for ML models
mdpi.com
). These ensure the workflow is grounded in current best practices.
Citations

Landsat Image Gallery - New York City Temperature and Vegetation
https://landsat.visibleearth.nasa.gov/view.php?id=6800
Favicon

Calculate and Classify Normalized Difference Results with EarthPy — EarthPy 0.9.4 documentation
https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_calculate_classify_ndvi.html
Favicon

Machine Learning for Simulation of Urban Heat Island Dynamics Based on Large-Scale Meteorological Conditions
https://www.mdpi.com/2225-1154/11/10/200

OSMnx: Python for Street Networks – Geoff Boeing
https://geoffboeing.com/2016/11/osmnx-python-street-networks/
Favicon

NDBI—ArcGIS Pro | Documentation
https://pro.arcgis.com/en/pro-app/latest/arcpy/spatial-analyst/ndbi.htm
Favicon

Machine Learning for Urban Heat Island (UHI) Analysis
https://www.sciencedirect.com/science/article/abs/pii/S2212095524001585

ARSET - Satellite Remote Sensing for Measuring Urban Heat Islands and Constructing Heat Vulnerability Indices | NASA Applied Sciences
https://appliedsciences.nasa.gov/get-involved/training/english/arset-satellite-remote-sensing-measuring-urban-heat-islands-and

Land Surface Temperature MODIS Visualization (LaSTMoV)(LAR-18877-1) | NASA Software Catalog
https://software.nasa.gov/software/LAR-18877-1
All Sources
landsat....arth.nasa
Faviconearthpy.readthedocs
Faviconmdpi
geoffboeing
Faviconpro.arcgis
Faviconsciencedirect
