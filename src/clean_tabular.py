import pandas as pd
import numpy as np
import seaborn as sns

def clean_categorical(df, clm, threshold, one_hot = True):
    value_counts = df[clm].value_counts()
    keep = value_counts[value_counts > threshold].index
    df[clm] = df[clm].apply(lambda x: x if x in keep else 'other')
    if one_hot:
        one_hot = pd.get_dummies(df[clm])
        try:
            df = df.join(one_hot)
        except ValueError:
            df = df.join(one_hot, lsuffix = '_clm')
        df = df.drop([clm, 'other'],axis = 1)
    else:
        vc = df[clm].value_counts()
        df[clm] = df[clm].apply(lambda x: vc.loc[x] if type(x) == str else np.NaN)
    return df

def clean(df):
    # drop all the useless data
    good_columns = ['BLDG_VAL', 'LAND_VAL', 'OTHER_VAL',
           'LOT_SIZE', 'LS_DATE', 'LS_PRICE', 'USE_CODE',
           'ZONE', 'YEAR_BUILT', 'BLD_AREA',
           'UNITS', 'RES_AREA', 'STYLE', 'STORIES', 'NUM_ROOMS', 'LOT_UNITS', 'MBL']
    df = df[good_columns]

    # convert fake missing values (0s and 1s) to NaN
    zero_nan_columns = ['BLDG_VAL', 'LAND_VAL', 'OTHER_VAL', 'LS_PRICE', 'LOT_SIZE', 'STORIES', 'NUM_ROOMS','YEAR_BUILT','BLD_AREA','UNITS','RES_AREA','LOT_UNITS']
    df[zero_nan_columns] = df[zero_nan_columns].replace(0, np.nan)

    one_nan_columns = ['LS_PRICE']
    df[one_nan_columns] = df[one_nan_columns].replace(1, np.nan)

    # add is missing columns (columns chosen based on correlations of missing values)
    is_missing_column = ['BLD_AREA','BLDG_VAL','LOT_SIZE']
    is_missing_column_names = [clm+'_MISSING' for clm in is_missing_column]
    df[is_missing_column_names] = df[is_missing_column].isna().astype('int32')

    # convert last sale date to just year
    df.LS_YEAR = df['LS_DATE'].apply(lambda x: x.year)
    df = df.drop('LS_DATE', axis = 1)

    #clean use codes
    df = clean_categorical(df, 'USE_CODE', 1)

    #clean ZONE
    df = clean_categorical(df, 'ZONE', 1)

    # clean STYLE
    df = clean_categorical(df, 'STYLE', 1)

    # style_mapping = {
    #  'MULTIFAMILY': ['Condominium', 'Apartments', 'Conventional-Apts','Mansard-Apts','Mid rise','Low rise','High Rise Apt','Mid Rise Apartments','Victorian-Apts'],
    #  '3_FLOORS':['3-Decker', 'Three decker','3-Decker-Apts'],
    #  'TRIPLEX':['3 fam Conv'],
    #  '2_FLOOR':['2-Decker', 'Two decker','2-Decker-Apts'],
    #  'DUPLEX':['Family Duplex', '2 Fam Conv', 'Duplex', 'Two Family','Two Family-Apts','Family Duplex-Apts'],
    #  'CONVENTIONAL':['Conventional', 'Fam Conv', 'Mansard'],
    #  'OTHER':['Stores/Apt Com'],
    #  'ROW_HOME':['Row End', 'Row Mid','Row End-Apts', 'Row Mid-Apts','Row Middle'],
    #  'TOWNHOUSE':['Townhouse end','Townhouse middle','Townhouse'],
    #  'VICTORIAN':['Victorian'],
    #  'COTTAGE':['Cottage Bungalow','Cottage']
    # }
    #
    # style_mapping_long = {}
    # for style_category in style_mapping:
    #     for style in style_mapping[style_category]:
    #         style_mapping_long[style] = style_category
    # df['STYLE'] = df['STYLE'].apply(lambda x: style_mapping_long[x])
    # df = df.drop('STYLE', axis = 1)

    # add distance to neighbors
    building_dist = pd.read_csv('./data/buildings_clean.csv', index_col = 0)
    df = df.merge(building_dist[['MBL', '1ST_CLOSEST', '2ND_CLOSEST']], how='left')

    # add thresholds
    df['LARGE_2ND_CLOSEST'] = (df['2ND_CLOSEST'] > 30).astype('int')
    df['BUILT_AFTER_1950'] = (df['YEAR_BUILT'] > 1950).astype('int')

    # add building to lot ratio
    df['BUILDING_LOT_RATIO'] = df.BLD_AREA/df.LOT_SIZE

    # add building parcel geometric features
    building_parcel_geometry = pd.read_csv('./data/building_parcel_geometric_features.csv', index_col=0)
    df = df.merge(building_parcel_geometry, how = 'left')

    # add parking permit value_counts
    parking_permit_counts = pd.read_csv('./data/parking_permit_counts.csv', index_col = 0)
    df = df.merge(parking_permit_counts, how = 'left')

    # add assessment data
    assessor = pd.read_csv('./data/assessor_clean.csv', index_col = 0)
    df = df.merge(assessor, how = 'left')

    assessor_tabular = [
        'ROOF_STRUCTURE_DESCRIP',
        'ROOF_COVER_DESCRIP',
        'INT_WALL_1_DESCRIP',
        'HEAT_TYPE',
        'FUEL_TYPE',
        'AC_TYPE',
        'GRADE_DESCRIP'
    ]

    for clm in assessor_tabular:
        df = clean_categorical(df, clm, 2)

    # impute missing values with mean
    df = df.fillna(df.mean())

    # take max over multiple units per MBL
    df = df.groupby('MBL').mean().reset_index()

    # one more time, for good luck
    df = df.fillna(df.mean())

    print(df.isnull().values.any())
    print(df.isna().values.any())

    return df


if __name__ == '__main__':
    #load assessor data from excel spreadsheet
    df = pd.read_excel('./data/residence_addresses_googlestreetview.xlsx',
                       sheet_name='residential by unit')
    df = clean(df)
    df.to_csv('./data/residence_addresses_googlestreetview_clean.csv')
