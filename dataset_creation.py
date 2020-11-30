# Procesamiento de datos.

# Importar archivos, para trabajarlos con Pandas.
from pathlib import Path
import pandas as pd
# Normalización de datos:
import sklearn.preprocessing as skp

banned_features = [
   'mean_mag_1',
   'mean_mag_2',
   'min_mag_1',
   'min_mag_2',
   'Mean_1',
   'Mean_2',
   'n_det_1',
   'n_det_2',
   'n_pos_1',
   'n_pos_2',
   'n_neg_1',
   'n_neg_2',
   'first_mag_1',
   'first_mag_2',
   'MHPS_non_zero_1',
   'MHPS_non_zero_2',
   'MHPS_PN_flag_1',
   'MHPS_PN_flag_2',
   #'W1', 'W2', 'W3', 'W4',
   'iqr_1',
   'iqr_2',
   'delta_mjd_fid_1',
   'delta_mjd_fid_2',
   'last_mjd_before_fid_1',
   'last_mjd_before_fid_2'#,
   #'g-r_ml',
   #'MHAOV_Period_1', 'MHAOV_Period_2'
]

def get_dataset(Joaquin):
    if Joaquin:
        etiquetas_dir = "/Users/joaquincubelli/Desktop/Inteligencia Computacional/ALeRCE_data/Etiquetas/dfcrossmatches_prioritized_v7.0.1.csv" # csv
        features_dir = "/Users/joaquincubelli/Desktop/Inteligencia Computacional/ALeRCE_data/storage/ztf_workspace/historic_data_20200916/features_20200916.parquet" # parquet

    else:
        etiquetas_dir = "/media/dela/1TB/A Universidad/Electrica/VIII Sem/Inteligencia Computacional/AlErCe/dfcrossmatches_prioritized_v7.0.1.csv" # csv
        features_dir = "/media/dela/1TB/A Universidad/Electrica/VIII Sem/Inteligencia Computacional/AlErCe/features/storage/ztf_workspace/historic_data_20200916/features_20200916.parquet" # parquet
        
    etiquetas = pd.read_csv(etiquetas_dir)
    etiquetas = etiquetas[['oid','classALeRCE']]
    
    features = pd.read_parquet(features_dir)
    features = features.drop(banned_features, axis=1)

    dataset = pd.merge(features, etiquetas, left_on='index', right_on='oid').drop(['oid'], axis=1)

    dataset_num = dataset.drop(['index','classALeRCE'], axis=1)
    norm_func = skp.QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
    dataset_norm = pd.DataFrame(norm_func.fit_transform(dataset_num), columns = dataset_num.columns)

    dataset[dataset_norm.columns] = dataset_norm
    dataset = dataset.fillna(-1)

    Dict = {
       "AGN": "AGN",
       "Blazar": "Blazar",
       "CV/Nova": "CV/Nova",
       "SNIa": "SNIa",
       "SNIbc": "SNIbc",
       "SNII": "SNII",
       "SNIIn": "SNII",
       "SLSN": "SLSN",
       "EA": "E",
       "EB/EW": "E",
       "DSCT": "DSCT",
       "RRL": "RRL",
       "Ceph": "CEP",
       "LPV": "LPV",
       "Periodic-Other": "Periodic-Other",
       "QSO": "QSO",
       "YSO": "YSO",
       "RSCVn": "Periodic-Other"
     }

    dataset = dataset.replace(to_replace = Dict)

    ticks = []
    for et in dataset['classALeRCE']:
        if et not in ticks:
            ticks.append(et)

    classes = []
    for x, y in Dict.items():
        classes.append(y)

    dataset = dataset[dataset.classALeRCE.isin(classes)]

    dataset.to_csv('../ALeRCE.csv')

    return dataset
