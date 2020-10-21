import pandas as pd
from sklearn.cluster import KMeans
    
    
def cluster_features(X_train, X_validate, X_test):   
    cluster_columns_1 = ['longitude_scaled',
                         'latitude_scaled',
                         'structure_tax_scaled']

    cluster_columns_2 = ['has_basement_scaled',
                         'has_hottub_or_spa_scaled',
                         'has_pool_scaled',
                         'has_patio_scaled',
                         'has_shed_scaled']

    cluster_columns_3 = ['livable_area_sqft_scaled',
                         'lot_size_sqft_scaled',
                         'year_built_scaled']

    kmeans1 = KMeans(n_clusters=7, random_state=0)
    kmeans2 = KMeans(n_clusters=6, random_state=0)
    kmeans3 = KMeans(n_clusters=6, random_state=0)

    cluster_1 = X_train[cluster_columns_1].copy()
    cluster_2 = X_train[cluster_columns_2].copy()
    cluster_3 = X_train[cluster_columns_3].copy()

    X_train['cluster_1'] = kmeans1.fit(cluster_1).labels_
    X_validate['cluster_1'] = kmeans1.predict(X_validate[cluster_columns_1])
    X_test['cluster_1'] = kmeans1.predict(X_test[cluster_columns_1])

    X_train['cluster_2'] = kmeans2.fit(cluster_2).labels_
    X_validate['cluster_2'] = kmeans2.predict(X_validate[cluster_columns_2])
    X_test['cluster_2'] = kmeans2.predict(X_test[cluster_columns_2])

    X_train['cluster_3'] = kmeans3.fit(cluster_3).labels_
    X_validate['cluster_3'] = kmeans3.predict(X_validate[cluster_columns_3])
    X_test['cluster_3'] = kmeans3.predict(X_test[cluster_columns_3])
    
    return X_train, X_validate, X_test, cluster_1, cluster_2, cluster_3
    
    
# df = prepare_zillow()
# train, validate, test = train_validate_test(df)


# geo_features = ['fips', 'latitude', 'longitude', 'logerror']


# home_features = ['parcelid',
#                  'num_of_bedrooms',
#                  'num_of_restrooms',
#                  'living_room_area_sqft',
#                  'lot_size_sqft',
#                  'year_built',
#                  'has_basement',
#                  'has_hottub_or_spa',
#                  'has_pool',
#                  'pool_area_sqft',
#                  'has_patio',
#                  'patio_area_sqft',
#                  'has_shed',
#                  'basement_area_sqft',
#                  'logerror']

# tax_features = ['property_tax',
#                 'structure_tax',
#                 'land_tax',
#                 'taxable_value',
#                 'logerror']

# # create functions for plots
# # filter our dataset for geographic features
# geo_dists = distributions[geo_features].copy()
# for i in geo_dists.columns[:-1]:
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,5))
#     fig.subplots_adjust(left=None, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.2)
#     ax1.hist(geo_dists[i], bins=50)
#     ax2.scatter(geo_dists.logerror, geo_dists[i])
#     ax1.title.set_text(f'Distribution of {i}')
#     ax2.title.set_text(f'Log error and {i}')
#     ax2.set_xlabel('Log error')
#     ax2.set_ylabel(f'{i}')
#     plt.tight_layout()
#     plt.show()
    
# home_dists = distributions[home_features].copy()   
# for i in home_dists.columns[:-1]:
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,5))
#     fig.subplots_adjust(left=None, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.2)
#     ax1.hist(home_dists[i], bins=50)
#     ax2.scatter(home_dists.logerror, home_dists[i])
#     ax1.title.set_text(f'Distribution of {i}')
#     ax2.title.set_text(f'Log error and {i}')
#     ax2.set_xlabel('Log error')
#     ax2.set_ylabel(f'{i}')
#     plt.tight_layout()
#     plt.show()
    
# tax_dists = distributions[tax_features].copy()  
# for i in tax_dists.columns[:-1]:
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,5))
#     fig.subplots_adjust(left=None, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
#     ax1.hist(tax_dists[i], bins=50)
#     ax2.scatter(tax_dists.logerror, tax_dists[i])
#     ax1.title.set_text(f'Distribution of {i}')
#     ax2.title.set_text(f'Log error and {i}')
#     ax2.set_xlabel('Log error')
#     ax2.set_ylabel(f'{i}')
#     plt.tight_layout()
#     plt.show()
