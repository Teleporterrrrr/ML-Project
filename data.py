import pandas as pd
import numpy as np
from scipy import stats
import math
import torch
from torch.nn.utils.rnn import pad_sequence


def load_data(path):
    data = pd.read_csv(path, header=0, index_col=0)
    #print(data)
    return data

def mode(x):
    return stats.mode(x)[0][0]

def split_data(x, y, split=0.8):
    idx = np.random.permutation(x.index)
    split_idx = int(split * len(x))
    train_idx, test_idx = idx[:split_idx], idx[split_idx:]

    train_x, test_x = x.loc[train_idx], x.loc[test_idx]

    train_ids = train_x["patientunitstayid"].unique()
    test_ids = test_x["patientunitstayid"].unique()
    train_y, test_y = y.loc[y["patientunitstayid"].isin(train_ids)], y.loc[y["patientunitstayid"].isin(test_ids)]

    return train_x, train_y, test_x, test_y

def preprocess_count(df):
    df = df[df["offset"] <= 7200]

    df["patientunitstayid"] = df["patientunitstayid"].astype('int32')

    ids = df["patientunitstayid"].unique()

    nursing_types = df["nursingchartcelltypevalname"].unique()

    nursing_types = [nursing for nursing in nursing_types if str(nursing) != 'nan']


    col_names = ["patientunitstayid", "lengthofstay", "admissionheight", "admissionweight", "age", "ethnicity", "gender"]

    for val in nursing_types:  
        col_names.append(val)

    df.loc[df['age'] == '> 89', 'age'] = 90
    df['age'] = df['age'].astype(float)
    labname = df["labname"].unique()
    labname = [lab for lab in labname if str(lab) != 'nan']

    for val in labname:
        col_names.append(val)

    col_names = [x for x in col_names if str(x) != 'nan']

    col_vals = []

    col_vals.append(ids.tolist())
    col_vals.append(df.groupby(['patientunitstayid'])['offset'].max().tolist())
    col_vals.append(df.groupby(['patientunitstayid'])['admissionheight'].max().tolist())
    col_vals.append(df.groupby(['patientunitstayid'])['admissionweight'].max().tolist())
    col_vals.append(df.groupby(['patientunitstayid'])['age'].max().tolist())
    col_vals.append(df.groupby(['patientunitstayid'])['ethnicity'].agg(mode).tolist())
    col_vals.append(df.groupby(['patientunitstayid'])['gender'].agg(mode).tolist())

    grouped = df.groupby(['patientunitstayid',"nursingchartcelltypevalname"], as_index=False).size()
    grouped = grouped.pivot_table("size", "patientunitstayid", "nursingchartcelltypevalname")
    grouped = grouped.rename_axis(None, axis=1).reset_index()

    lab_grouped = df.groupby(['patientunitstayid', "labname"], as_index=False).size()
    lab_grouped = lab_grouped.pivot_table("size", "patientunitstayid", "labname")
    lab_grouped = lab_grouped.rename_axis(None, axis=1).reset_index()

    processed_df = pd.DataFrame(col_vals)
    processed_df = processed_df.T
    processed_df.columns = col_names[0:7]

    processed_df = pd.merge(processed_df, grouped, on="patientunitstayid", how="outer")
    processed_df = pd.merge(processed_df, lab_grouped, on="patientunitstayid", how="outer")

    processed_df[processed_df.columns.intersection(col_names[7:])] = processed_df[processed_df.columns.intersection(col_names[7:])].fillna(0)
    processed_df["ethnicity"] = processed_df["ethnicity"].astype("category")
    processed_df["gender"] = processed_df["gender"].astype("category")
    processed_df["lengthofstay"] = processed_df["lengthofstay"].astype("float")
    processed_df["admissionheight"] = processed_df["admissionheight"].astype("float")
    processed_df["admissionweight"] = processed_df["admissionweight"].astype("float")
    processed_df["age"] = processed_df["age"].astype("float")

    processed_df.drop(["ethnicity", "gender", "lengthofstay", "admissionheight", "admissionweight", "age"], axis=1, inplace=True)
    return processed_df

def preprocess_average(df):
    dtypes = {
        'patientunitstayid': np.int64,
        'admissionheight': float,
        'admissionweight': float,
        'age': str,
        'cellattributevalue': str,
        'celllabel': str,
        'ethnicity': str,
        'gender': str,
        'labmeasurenamesystem': str,
        'labname': str,
        'labresult': float,
        'nursingchartcelltypevalname': str,
        'nursingchartvalue': str,
        'offset': float,
        'unitvisitnumber': float
    }

    df.dropna(subset=['patientunitstayid'], inplace = True)

    comp_df = df.groupby('patientunitstayid').apply(lambda x: pd.Series({c: ','.join(x[c].astype(str).values) 
                                                                         for c in x.columns}))
    df_2 = comp_df[['patientunitstayid']]
    for df_column in df.columns:
        if df_column == 'patientunitstayid':
            continue
        df_2.loc[:, df_column] = comp_df[df_column].astype(str).apply(lambda x: ','.join([column for column in x.split(',')
                                                                            if column != 'nan' and column != '']))

    df_2 = df_2.reindex(columns=(['patientunitstayid'] + list(df.columns[~df.columns.isin(['patientunitstayid'])])))
    # df_2.to_csv('preprocess_.csv', index = False)
    df_2['Offset Max'] = np.nan
    df_2['Offset Min'] = np.nan
    df_2['Offset Total'] = np.nan
    df_2['O2 Saturation Avg'] = np.nan
    df_2['O2 Saturation Max'] = np.nan
    df_2['O2 Saturation Min'] = np.nan
    df_2['O2 Saturation RoC'] = np.nan
    df_2['Non-Invasive BP Mean Avg'] = np.nan
    df_2['Non-Invasive BP Mean Max'] = np.nan
    df_2['Non-Invasive BP Mean Min'] = np.nan
    df_2['Non-Invasive BP Mean RoC'] = np.nan
    df_2['Non-Invasive BP Systolic Avg'] = np.nan
    df_2['Non-Invasive BP Systolic Max'] = np.nan
    df_2['Non-Invasive BP Systolic Min'] = np.nan
    df_2['Non-Invasive BP Systolic RoC'] = np.nan
    df_2['Non-Invasive BP Diastolic Avg'] = np.nan
    df_2['Non-Invasive BP Diastolic Max'] = np.nan
    df_2['Non-Invasive BP Diastolic Min'] = np.nan
    df_2['Non-Invasive BP Diastolic RoC'] = np.nan
    df_2['Heart Rate Avg'] = np.nan
    df_2['Heart Rate Max'] = np.nan
    df_2['Heart Rate Min'] = np.nan
    df_2['Heart Rate RoC'] = np.nan
    df_2['GCS Total Avg'] = np.nan
    df_2['GCS Total Max'] = np.nan
    df_2['GCS Total Min'] = np.nan
    df_2['GCS Total RoC'] = np.nan
    df_2['Respiratory Rate Avg'] = np.nan
    df_2['Respiratory Rate Max'] = np.nan
    df_2['Respiratory Rate Min'] = np.nan
    df_2['Respiratory Rate RoC'] = np.nan
    df_2['Capillary Refill Normal'] = np.nan
    df_2['Capillary Refill < 2'] = np.nan
    df_2['Capillary Refill > 2'] = np.nan
    df_2['Glucose (mg/dL) Avg'] = np.nan
    df_2['Glucose (mg/dL) Max'] = np.nan
    df_2['Glucose (mg/dL) Min'] = np.nan
    df_2['Glucose (mg/dL) RoC'] = np.nan
    df_2['pH Avg'] = np.nan
    df_2['pH Max'] = np.nan
    df_2['pH Min'] = np.nan
    df_2['pH RoC'] = np.nan
    # print(df_2.columns)

    #################################################PATIENTUNITSTAYID
    for i, row in df_2.iterrows():
        p_list = df_2.at[i, 'patientunitstayid'].split(',')
        df_2.at[i, 'patientunitstayid'] = p_list[0]
    # print(df_2)

    #################################################PATIENTUNITSTAYID



    ################################################NURSINGCHART
    for i, row in df_2.iterrows():
        n_list = df_2.at[i, 'nursingchartcelltypevalname'].split(',')
        v_list = df_2.at[i, 'nursingchartvalue'].split(',')
        # print(row, len(n_list), len(v_list))
        new_list = []
        n2_list = []
        for ncval in range(len(v_list)):
            try:
                float_val = float(v_list[ncval])
                new_list.append(float_val)
                n2_list.append(n_list[ncval])
            except ValueError:
                pass
        v_list = new_list
        n_list = n2_list
        df_2.at[i, 'nursingchartcelltypevalname'] = np.nan
        df_2.at[i, 'nursingchartvalue'] = np.nan
        # print(n_list, v_list)
        if n_list:
            o2sat = []
            bpmean = []
            bpsys = []
            bpdy = []
            hr = []
            gcs = []
            rr = []

            for j in range(len(n_list)):
                if n_list[j] == 'O2 Saturation':
                    o2sat.append(v_list[j])
                    continue
                elif n_list[j] == 'Non-Invasive BP Mean':
                    bpmean.append(v_list[j])
                    continue
                elif n_list[j] == 'Non-Invasive BP Systolic':
                    bpsys.append(v_list[j])
                    continue
                elif n_list[j] == 'Non-Invasive BP Diastolic':
                    bpdy.append(v_list[j])
                    continue
                elif n_list[j] == 'Heart Rate':
                    hr.append(v_list[j])
                    continue
                elif n_list[j] == 'GCS Total':
                    gcs.append(v_list[j])
                    continue
                elif n_list[j] == 'Respiratory Rate':
                    rr.append(v_list[j])
                    continue

        if len(o2sat) != 0:
            o2sat_avg = sum(o2sat)/len(o2sat)
            o2sat_min = min(o2sat)
            o2sat_max = max(o2sat)
            o2sat_roc = (max(o2sat) - min(o2sat)) / (2)
        else:
            o2sat_avg = np.nan
            o2sat_min = np.nan
            o2sat_max = np.nan
            o2sat_roc = np.nan

        if len(bpmean) != 0:
            bpmean_avg = sum(bpmean)/len(bpmean)
            bpmean_min = min(bpmean)
            bpmean_max = max(bpmean)
            bpmean_roc = (max(bpmean) - min(bpmean)) / (2)
        else:
            bpmean_avg = np.nan
            bpmean_min = np.nan
            bpmean_max = np.nan
            bpmean_roc = np.nan

        if len(bpsys) != 0:
            bpsys_avg = sum(bpsys)/len(bpsys)
            bpsys_min = min(bpsys)
            bpsys_max = max(bpsys)
            bpsys_roc = (max(bpsys) - min(bpsys)) / (2)
        else:
            bpsys_avg = np.nan
            bpsys_min = np.nan
            bpsys_max = np.nan
            bpsys_roc = np.nan

        if len(bpdy) != 0:
            bpdy_avg = sum(bpdy)/len(bpdy)
            bpdy_min = min(bpdy)
            bpdy_max = max(bpdy)
            bpdy_roc = (max(bpdy) - min(bpdy)) / (2)
        else:
            bpdy_avg = np.nan
            bpdy_min = np.nan
            bpdy_max = np.nan
            bpdy_roc = np.nan

        if len(hr) != 0:
            hr_avg = sum(hr)/len(hr)
            hr_min = min(hr)
            hr_max = max(hr)
            hr_roc = (max(hr) - min(hr)) / (2)
        else:
            hr_avg = np.nan
            hr_min = np.nan
            hr_max = np.nan
            hr_roc = np.nan

        if len(gcs) != 0:
            gcs_avg = sum(gcs)/len(gcs)
            gcs_min = min(gcs)
            gcs_max = max(gcs)
            gcs_roc = (max(gcs) - min(gcs)) / (2)
        else:
            gcs_avg = np.nan
            gcs_min = np.nan
            gcs_max = np.nan
            gcs_roc = np.nan
        if len(rr) != 0:
            rr_avg = sum(rr)/len(rr)
            rr_min = min(rr)
            rr_max = max(rr)
            rr_roc = (max(rr) - min(rr)) / (2)
        else:
            rr_avg = np.nan
            rr_min = np.nan
            rr_max = np.nan
            rr_roc = np.nan

        df_2.at[i, 'O2 Saturation Avg'] = o2sat_avg
        df_2.at[i, 'O2 Saturation Max'] = o2sat_min
        df_2.at[i, 'O2 Saturation Min'] = o2sat_max
        df_2.at[i, 'O2 Saturation RoC'] = o2sat_roc

        df_2.at[i, 'Non-Invasive BP Mean Avg'] = bpmean_avg
        df_2.at[i, 'Non-Invasive BP Mean Max'] = bpmean_min
        df_2.at[i, 'Non-Invasive BP Mean Min'] = bpmean_max
        df_2.at[i, 'Non-Invasive BP Mean RoC'] = bpmean_roc

        df_2.at[i, 'Non-Invasive BP Systolic Avg'] = bpsys_avg
        df_2.at[i, 'Non-Invasive BP Systolic Max'] = bpsys_min
        df_2.at[i, 'Non-Invasive BP Systolic Min'] = bpsys_max
        df_2.at[i, 'Non-Invasive BP Systolic RoC'] = bpsys_roc
                                                    

        df_2.at[i, 'Non-Invasive BP Diastolic Avg'] = bpdy_avg
        df_2.at[i, 'Non-Invasive BP Diastolic Max'] = bpdy_min
        df_2.at[i, 'Non-Invasive BP Diastolic Min'] = bpdy_max
        df_2.at[i, 'Non-Invasive BP Diastolic RoC'] = bpdy_roc

        df_2.at[i, 'Heart Rate Avg'] = hr_avg
        df_2.at[i, 'Heart Rate Max'] = hr_min
        df_2.at[i, 'Heart Rate Min'] = hr_max
        df_2.at[i, 'Heart Rate RoC'] = hr_roc

        df_2.at[i, 'GCS Total Avg'] = gcs_avg
        df_2.at[i, 'GCS Total Max'] = gcs_min
        df_2.at[i, 'GCS Total Min'] = gcs_max
        df_2.at[i, 'GCS Total RoC'] = gcs_roc

        df_2.at[i, 'Respiratory Rate Avg'] = rr_avg
        df_2.at[i, 'Respiratory Rate Max'] = rr_min
        df_2.at[i, 'Respiratory Rate Min'] = rr_max
        df_2.at[i, 'Respiratory Rate RoC'] = rr_roc
        
    #################################################NURSINGCHART

    ################################################CELLATTRIBUTETYPE/CELLLABELS
    for i, row in df_2.iterrows():
        n_list = df_2.at[i, 'celllabel'].split(',')
        v_list = df_2.at[i, 'cellattributevalue'].split(',')
        # print(row, len(n_list), len(v_list))
        norm = 0
        less = 0
        gtr = 0
        total = 0
        for ncval in range(len(v_list)):
            if v_list[ncval] == "normal":
                norm += 1#############################
                total += 1##############################
            elif v_list[ncval] == "> 2 seconds":
                gtr += 1
                total += 1#############################
            elif v_list[ncval] == "< 2 seconds":
                less += 1
                total += 1##############################
            else:
                continue
        df_2.at[i, 'cellattributevalue'] = np.nan
        df_2.at[i, 'celllabel'] = np.nan

        if total != 0:
            caprefil_norm = norm/total
            caprefil_less = less/total
            caprefil_gtr = gtr/total

        else:
            caprefil_norm = np.nan
            caprefil_less = np.nan
            caprefil_gtr = np.nan

        df_2.at[i, 'Capillary Refill Normal'] = caprefil_norm
        df_2.at[i, 'Capillary Refill < 2'] = caprefil_less
        df_2.at[i, 'Capillary Refill > 2'] = caprefil_gtr
    #################################################CELLATTRIBUTETYPE/CELLLABEL

    ################################################LABNAME/LABRESULT
    for i, row in df_2.iterrows():
        n_list = df_2.at[i, 'labname'].split(',')
        v_list = df_2.at[i, 'labresult'].split(',')
        # print(row, len(n_list), len(v_list))
        new_list = []
        n2_list = []
        for ncval in range(len(v_list)):
            try:
                float_val = float(v_list[ncval])
                new_list.append(float_val)
                n2_list.append(n_list[ncval])
            except ValueError:
                pass
        v_list = new_list
        n_list = n2_list
        df_2.at[i, 'labname'] = np.nan
        df_2.at[i, 'labresult'] = np.nan
        # print(n_list, v_list)
        glucose = []
        pH = []
        if n_list:
            for j in range(len(n_list)):
                if n_list[j] == 'glucose':
                    glucose.append(v_list[j])
                    continue
                elif n_list[j] == 'pH':
                    pH.append(v_list[j])
                    continue

        if len(glucose) != 0:
            glucose_avg = sum(glucose)/len(glucose)
            glucose_max = max(glucose)
            glucose_min = min(glucose)
            glucose_roc = (max(glucose) - min(glucose)) / 2
        else:
            glucose_avg = np.nan
            glucose_max = np.nan
            glucose_min = np.nan
            glucose_roc = np.nan
        if len(pH) != 0:
            ph_avg = sum(pH)/len(pH)
            ph_max = max(pH)
            ph_min = min(pH)
            ph_roc = (max(pH) - min(pH)) / 2
        else:
            ph_avg = np.nan
            ph_max = np.nan
            ph_min = np.nan
            ph_roc = np.nan

        df_2.at[i, 'Glucose (mg/dL) Avg'] = glucose_avg
        df_2.at[i, 'Glucose (mg/dL) Max'] = glucose_max
        df_2.at[i, 'Glucose (mg/dL) Min'] = glucose_min
        df_2.at[i, 'Glucose (mg/dL) RoC'] = glucose_roc

        df_2.at[i, 'pH Avg'] = ph_max
        df_2.at[i, 'pH Max'] = ph_max
        df_2.at[i, 'pH Min'] = ph_min
        df_2.at[i, 'pH RoC'] = ph_roc
    #################################################LABNAME/LABRESULT
    
    #################################################OFFSET
    for i, row in df_2.iterrows():
        off_list = df_2.at[i, 'offset'].split(',')
        off_list = [float(offset) for offset in off_list if offset]
        if len(off_list) != 0:###################################
            offset_total = (max(off_list) - min(off_list))
            offset_max = max(off_list)
            offset_min = min(off_list)
        else:
            offset_total = np.nan
            offset_max = np.nan
            offset_min = np.nan

        df_2.at[i, 'Offset Total'] = offset_total
        df_2.at[i, 'Offset Max'] = offset_max
        df_2.at[i, 'Offset Min'] = offset_min
    #################################################OFFSET

    dtypes = {
        'patientunitstayid': np.int64,
        'admissionheight': float,
        'admissionweight': float,
        'age': str,
        'ethnicity': str,
        'gender': str,
        'unitvisitnumber': float,
        'Offset Total': float,
        'Offset Max': float,
        'Offset Min': float,
        'O2 Saturation Avg': float,
        'O2 Saturation Max': float,
        'O2 Saturation Min': float,
        'O2 Saturation RoC': float,
        'Non-Invasive BP Mean Avg': float,
        'Non-Invasive BP Mean Max': float,
        'Non-Invasive BP Mean Min': float,
        'Non-Invasive BP Mean RoC': float,
        'Non-Invasive BP Systolic Avg': float,
        'Non-Invasive BP Systolic Max': float,
        'Non-Invasive BP Systolic Min': float,
        'Non-Invasive BP Systolic RoC': float,
        'Non-Invasive BP Diastolic Avg': float,
        'Non-Invasive BP Diastolic Max': float,
        'Non-Invasive BP Diastolic Min': float,
        'Non-Invasive BP Diastolic RoC': float,
        'Heart Rate Avg': float,
        'Heart Rate Max': float,
        'Heart Rate Min': float,
        'Heart Rate RoC': float,
        'GCS Total Avg': float,
        'GCS Total Max': float,
        'GCS Total Min': float,
        'GCS Total RoC': float,
        'Respiratory Rate Avg': float,
        'Respiratory Rate Max': float,
        'Respiratory Rate Min': float,
        'Respiratory Rate RoC': float,
        'Capillary Refill Normal': float,
        'Capillary Refill less than 2': float,
        'Capillary Refill greater than 2': float,
        'Glucose (mg/dL) Avg': float,
        'Glucose (mg/dL) Max': float,
        'Glucose (mg/dL) Min': float,
        'Glucose (mg/dL) RoC': float,
        'pH Avg': float,
        'pH Max': float,
        'pH Min': float,
        'pH RoC': float
    }
    df_2.rename(columns={'Capillary Refill < 2':'Capillary Refill less than 2',
                         'Capillary Refill > 2':'Capillary Refill greater than 2'}, inplace=True)
    df_2 = df_2.drop('offset', axis = 1)
    df_2 = df_2.drop('nursingchartcelltypevalname', axis = 1)
    df_2 = df_2.drop('nursingchartvalue', axis = 1)
    df_2 = df_2.drop('cellattributevalue', axis = 1)
    df_2 = df_2.drop('celllabel', axis = 1)
    df_2 = df_2.drop('labmeasurenamesystem', axis = 1)
    df_2 = df_2.drop('labname', axis = 1)
    df_2 = df_2.drop('labresult', axis = 1)
    for df_column in df_2.columns:
        if df_2[df_column].dtype in ['float64', np.int64]:
            df_2[df_column] = df_2[df_column].astype(dtypes[df_column])
    df_2 = df_2.replace('', np.nan)
    df_2.fillna(method = 'ffill', inplace = True)
    df_2.loc[df_2['age'] == '> 89', 'age'] = 90
    df_2['age'] = df_2['age'].astype(np.int64)
    df_2['ethnicity'] = df_2['ethnicity'].astype("category")
    df_2['gender'] = df_2['gender'].astype("category")

    df_2.reset_index(drop=True, inplace=True)
    print(df_2)
    # print(df_2['unitvisitnumber'].dtype)
    
    return df_2

def preprocess_x(df):
    count_df = preprocess_count(df)
    average_df = preprocess_average(df)
    average_df.reset_index(drop=True, inplace=True)

    count_df['patientunitstayid'] = count_df['patientunitstayid'].astype(float)
    average_df['patientunitstayid'] = average_df['patientunitstayid'].astype(float)
    count_df.to_csv("count_df.csv")
    average_df.to_csv("average_df.csv")
    final_df = pd.merge(count_df, average_df, how="outer", on="patientunitstayid")
    
    final_df.to_csv("merged_df.csv", index=False)

    final_df['admissionheight'] = final_df['admissionheight'].astype(float)
    final_df['admissionweight'] = final_df['admissionweight'].astype(float)
    final_df['unitvisitnumber'] = final_df['unitvisitnumber'].astype(float)

    return final_df