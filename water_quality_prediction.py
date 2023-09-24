import numpy as np
import pandas as pd
from scipy.stats import mode

import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearnex import patch_sklearn
patch_sklearn()

import torch
import intel_extension_for_pytorch as ipex

from pytorch_tabnet.tab_model import TabNetClassifier

clf1 = TabNetClassifier()
clf1.load_model('./models/model_1.zip')

clf2 = TabNetClassifier()
clf2.load_model('./models/model_2.zip')

clf3 = TabNetClassifier()
clf3.load_model('./models/model_3.zip')

# data to be used for the final pipeline (obtained from preprocessing and data analysis)
drop_column_rows = ['Iron', 'Nitrate', 'Lead', 'Color', 'Turbidity', 'Odor', 'Chlorine', 'Total Dissolved Solids', 
                    'Source', 'Air Temperature', 'Month', 'Day', 'Time of Day']

fillna_cols = ['pH', 'Chloride', 'Zinc', 'Fluoride', 'Copper', 'Sulfate', 'Conductivity', 'Manganese', 'Water Temperature']
fillna_cols_means = [7.445251285427391, 184.3043772737876, 1.5504042963151854, 0.9647085346855051, 0.5162646133771908,
                     146.07352061294569, 425.01489236635155, 0.10923347178334412, 19.128374309638666]

sourceslist = ['Aquifer', 'Ground', 'Lake', 'Reservoir', 'River', 'Spring', 'Stream', 'Well']
replacesourceslist = [0, 1, 2, 3, 4, 5, 6, 7]

colorslist = ['Colorless', 'Near Colorless', 'Faint Yellow', 'Light Yellow', 'Yellow']
replacecolorslist = [0, 1, 2, 3, 4]


def FinalPipeline(data):
    '''
    Input: Dataframe to be classified
    
    Returns the prediction array based on the Ensembled TabNet Classifiers
    '''
    data = data.drop(['Index'], axis=1)
    data = data.fillna(np.nan)
    
    # ensure that the data does not have any null values 
    # otherwise some rows will get dropped
    for each in drop_column_rows:
        data = data[pd.notna(data[each])]
    
    for each in range(len(fillna_cols)):
        data[fillna_cols[each]].fillna(fillna_cols_means[each], inplace=True)

    data['Color'].replace(colorslist, replacecolorslist, inplace=True)
    data['Source'].replace(sourceslist, replacesourceslist, inplace=True)

    data = data.drop(['Month', 'Day', 'Time of Day'], axis=1)

    data = np.array(data)
    print(data.shape)

    y_preds = []
    for each in [clf1, clf2, clf3]:
        y_preds.append(each.predict(data))

    return mode(y_preds).mode.reshape(-1)
