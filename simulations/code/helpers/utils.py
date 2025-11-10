import numpy as np
import pandas as pd

print('Utils excecuted')


#CHECKING FUNCTION
def checking_function():
    return('Utils is working')


# CONVERT STRING COLUMNS OF DF TO LISTS
def convert_strings_to_lists(df, columns):
    """
    If the csv contains a column that is ',' separated, that column is read as a string.
    We want to convert that string to a list of values. We try to make the list float or string.
    """
    def tolist(stringvalue):
        if isinstance(stringvalue, str):
            try:
                stringvalue = stringvalue.split(sep=',')
                try:
                    val = np.array(stringvalue, dtype=float)
                except:
                    val = np.array(stringvalue)
            except:
                val = np.array([])
        elif np.isnan(stringvalue):
            return np.array([])
        else:
            val = np.array([stringvalue])
        return val.tolist()

    for column in columns:
        df[column] = df[column].apply(tolist)
    return df


# UNNESTING LISTS IN COLUMNS DATAFRAMES
def unnesting(df, explode):
    """
    Unnest columns that contain list creating a new row for each element in the list.
    The number of elements must be the same for all the columns, row by row.
    """
    length = df[explode[0]].str.len()
    idx = df.index.repeat(length)
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx
    finaldf = df1.join(df.drop(explode, 1), how='left')
    finaldf.reset_index(drop=True, inplace=True)

    length2 = [list(range(l)) for l in length]
    length2 = [item + 1 for sublist in length2 for item in sublist]
    name = explode[0] + '_index'
    finaldf[name] = length2
    return finaldf

# SUBJECT TAGS
def subjects_tags():
    '''Identifies the subject depending on the tag
     ECOHAB reads tags with reversed order by pairs'''
    all_subjects = ['man', 'T1', 'T2', 'T3',
                    'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A50', 'A51', 'A52']
    all_mv_tags = ['041A9DB979', '041A9C89B3', '041A9C7958', '0419A8212D',
                   '0417CA5FDE', '041A9DBD90', '0419A86ECB', '0419A8218D', '0417CA97FA', '0419A8701C',
                   '041A9D7BE0', '0419A822D2', '041A9DBDF9', '041A9DB349', '0419A81BFB', '041A9D86C5']
    all_colors = ['lightsteelblue', 'mediumseagreen', 'greenyellow', 'salmon',
              'yellow', 'orange', 'tomato', 'crimson', 'mediumvioletred',
              'darkorchid', 'darkblue', 'royalblue', 'lightskyblue', 'mediumaquamarine',
              'green', 'yellowgreen']

    all_ecohab_tags = []  # ECOHAB reads tags with reversed order by pairs
    for tag in all_mv_tags:  # loop thought MV tags
        tag_r = tag[::-1]  # revert
        new_tag = ''
        for (front, back) in zip(tag_r[0::2], tag_r[1::2]):  # invert 2 by 2
            new_tag += back + front
        all_ecohab_tags.append(new_tag)

    return all_subjects, all_ecohab_tags, all_colors

# BASAL WEIGHTS
def relative_weights(subject, weight):
    basal_weights = {
    'A5': '32.68', 'A6': '31.46', 'A7': '30.40', 'A8': '31.38', 'A9': '31.65', 'A10': '27.71', 'A11': '31.20', 'A12': '27.72',
    'MA1': '31.3', 'MA2': '25.9', 'MA3': '28.2', 'MA4': '27', 'MA5': '30.9',
    'A13':'23.4', 'A14':'21.63', 'A15':'21.8', 'A16':'21.87', 'A17':'22.7', 'A18':'21.37', 'A19':'23.7', 'A20':'24.1',
    'MA6': '24.84', 'MA7': '26.48', 'MA8': '27.51', 'MA9': '24', 'MA10': '25',
    'MA11': '24.84', 'MA12': '26.48', 'MA13': '27.51', 'MA14': '24', 'MA15': '25',
    'A21':'19.77', 'A22':'20.1', 'A23':'21.1', 'A24':'22.73', 'A25':'21.3','A26':'20.4', 'A27':'21.8','A28':'22.77', 'A29':'22.8', 'A30':'24.1',
    'A31':'21.9', 'A32':'22', 'A33':'22.1', 'A34':'26.6', 'A35':'22.5','A36':'23.2', 'A37':'21.7','A38':'22.3', 'A39':'22.6', 'A40':'21.6',
    'A41':'22.6', 'A42':'27.6', 'A43':'23.2', 'A44':'22.2', 'A45':'25.2','A46':'21.4', 'A47':'24.9','A48':'25.6', 'A49':'22.7', 'A50':'23.9',
    'A51':'23.5', 'A52':'27.2',
    'A53':'23.4', 'A54':'24.1', 'A55':'23.7','A56':'22.4', 'A57':'23.9','A58':'23.7', 'A59':'21.9', 'A60':'27.5',
    'A61':'23.9', 'A62':'26.6', 'A63':'28.4',
    'A64': '26.9', 'A65': '24.5', 'A66': '26.5', 'A67': '26.2', 'A68': '27.6',
    'A69': '21.4', 'A70': '21.9', 'A71': '23.8', 'A72': '21.7', 'A73': '24.2',
    'A74': '24.6', 'A75': '26.0', 'A76': '26.0', 'A77': '24.3',
    'A78':'27.9', 'A79':'26.9', 'A80':'26.1', 'A81':'25.1', 'A82':'27.2',
    'A83': '26.65', 'A84':'26.3', 'A85':'25.9', 'A86':'25.75', 'A87':'25.6', 'A88':'24.3', 'A89':'25.8', 'A90': '24.3',
    'A91':'24.7', 'A92':'24.7', 'A93':'22.4', 'A94':'22.5', 'A95':'22.5', 'A96':'22.5', 'A97':'21.5',
    'B1': '26.6', 'B2': '23.3', 'B3': '26.7', 'B4': '26.6', 'B5': '30.6', 'B6': '29.6', 'B7': '25.2', 'B8': '27.5',
    'B9': '28.3', 'B10': '28.0', 'B11': '25.7', 'B12': '24.4', 'B13': '25.8', 'B14': '26',
    'T1': '33' , 'T2':'32.7','T3': '32.6' , 'T4':'32.9', 'T5':'26.7','T6': '27.5' , 'T7':'27.7', 'T8':'22.6', 'T9':'30.9'}

    for key, value in basal_weights.items():
        if subject == key:
            basal_weight_subj = float(value)
            relative_weight_subj = weight / basal_weight_subj * 100
            return relative_weight_subj

# COMPUTE WINDOW AVERAGE
def compute_window(data, runningwindow):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    for i in range(len(data)):
        if i < runningwindow:
            performance.append(round(np.mean(data[0:i + 1]), 2))
        else:
            performance.append(round(np.mean(data[i - runningwindow:i]), 2))
    return performance


# COLLECT ALL REPONSES TIMES IN A COLUMN
def create_responses_time(row):
    try:
        result = row['STATE_Incorrect_START'].tolist().copy()
    except (TypeError, AttributeError):
        result = row['STATE_Incorrect_START'].copy()
    items = [row['STATE_Correct_first_START'], row['STATE_Correct_other_START'], row['STATE_Punish_START']]
    for item in items:
        if not np.isnan(item):
            result += [item]
    return result


# RESPONSE RESULT COLUMN
def create_reponse_result(row):
    result = ['incorrect'] * len(row['STATE_Incorrect_START'])
    if row['trial_result'] != 'miss' and row['trial_result'] != 'incorrect':
        result += [row['trial_result']]
    return result




# # ORDER LISTS
# def order_lists(list, type):
#     if type == 'ttypes':
#         order = ['VG', 'WM_I', 'WM_D', 'WM_Ds', 'WM_Dm', 'WM_Dl']
#         c_order = ['#393b79', '#6b6ecf', '#9c9ede', '#9c9ede', '#ce6dbd', '#a55194']
#     elif type == 'treslts':
#         order = ['correct_first', 'correct_other', 'punish', 'incorrect', 'miss']
#         c_order = ['green', 'limegreen', 'firebrick', 'red', 'black']
#     elif type == 'probs':
#         order = ['pvg', 'pwm_i', 'pwm_d', 'pwm_ds', 'pwm_dl']
#         c_order = ['#393b79', '#6b6ecf', '#9c9ede', '#9c9ede', '#a55194']
#
#     ordered_list = []
#     ordered_c_list = []
#
#     for idx, i in enumerate(order):
#         if i in list:
#             ordered_list.append(i)
#             ordered_c_list.append(c_order[idx])
#
#     return ordered_list, ordered_c_list
# ORDER LISTS
def order_lists(list, type):
    '''Returns ordered lists with the differnt trial types and its corresponding colors lists'''
    vg_c = 'MidnightBlue'
    ds_c = 'RoyalBlue'
    dm_c = 'CornflowerBlue'
    dl_c = 'LightSteelBlue'
    if type == 'ttypes':
        order = ['VG', 'DS', 'DSc1', 'DSc2', 'DM', 'DMc1', 'DL']
        c_order = [vg_c, ds_c, ds_c, ds_c, dm_c, dm_c, dl_c]
    elif type == 'treslts':
        order = ['correct_first', 'correct_other', 'punish', 'incorrect', 'miss']
        c_order = ['green', 'limegreen', 'firebrick', 'red', 'black']
    elif type == 'probs':
        order = ['pvg', 'pds', 'pdsc1', 'pdsc2', 'pdm', 'pdmc1', 'pdl']
        c_order = [vg_c, ds_c, ds_c, ds_c, dm_c, dm_c, dl_c]

    ordered_list = []
    ordered_c_list = []

    for idx, i in enumerate(order):
        if i in list:
            ordered_list.append(i)
            ordered_c_list.append(c_order[idx])
    return ordered_list, ordered_c_list


# STIMULUS CALCULATION
### warning last modification: try except in the function because row['task] was giving problems sometimes.
### important to modify form the code the way of calling the function:
### df['stim_onset'], df['stim_duration'], df['stim_offset'] = zip(*df.apply(lambda row: stimulus_duration_calculation(row), axis=1))

def stimulus_duration_calculation(row):
    ''' Calculates the stimulus onset, offset and duration.
        Extends stimulus duration adding extra time up to the maximum when necessary '''
    
    try:
        if 'StageTraining' in row['task']:

            if 'DS' in row['trial_type']:
                if row['trial_type'] == 'DS':
                    stim_onset = row['STATE_Fixation1_START']
                elif row['trial_type'] == 'DSc1':
                    stim_onset = row['STATE_Fixation3_START']
                elif row['trial_type'] == 'DSc2':
                    stim_onset = row['STATE_Fixation2_START']

                stim_offset = row['STATE_Fixation3_END']
                stim_duration = stim_offset - stim_onset

                if row['stim_dur_ds'] > 0:  # stimulus duration extended to the next state
                    stim_dur_ext = stim_duration + row['stim_dur_ds']
                    max_dur = row['response_window_end'] - stim_onset
                    if stim_dur_ext <= max_dur:  # extend when don't overcome max
                        stim_duration = stim_dur_ext
                    elif stim_dur_ext > max_dur:  # take the maximum when overcome
                        stim_duration = max_dur
                    stim_offset = stim_onset + stim_duration  # correct stimulus offset


            elif 'DM' in row['trial_type']:
                if row['trial_type'] == 'DM':
                    stim_onset = row['STATE_Fixation1_START']
                elif row['trial_type'] == 'DMc1':
                    stim_onset = row['STATE_Fixation2_START']

                stim_offset = row['STATE_Fixation2_END']
                stim_duration = stim_offset - stim_onset

                if row['stim_dur_dm'] > 0:  # stimulus duration extended to the next state
                    stim_dur_ext = stim_duration + row['stim_dur_dm']
                    max_dur = row['STATE_Fixation3_END'] - stim_onset
                    if stim_dur_ext <= max_dur:  # extend when don't overcome max
                        stim_duration = stim_dur_ext
                    elif stim_dur_ext > max_dur:  # take the maximum when overcome
                        stim_duration = max_dur
                    stim_offset = stim_onset + stim_duration  # correct stimulus offset


            elif 'DL' in row['trial_type']:
                stim_onset = row['STATE_Fixation1_START']
                stim_offset = row['STATE_Fixation1_END']
                stim_duration = stim_offset - stim_onset

                if row['stim_dur_dl'] > 0:  # stimulus duration extended to the next state
                    stim_dur_ext = stim_duration + row['stim_dur_dl']
                    max_dur = row['STATE_Fixation2_END'] - stim_onset
                    if stim_dur_ext <= max_dur:  # extend when don't overcome max
                        stim_duration = stim_dur_ext
                    elif stim_dur_ext > max_dur:  # take the maximum when overcome
                        stim_duration = max_dur
                    stim_offset = stim_onset + stim_duration  # correct stimulus offset

                    
            elif 'VG' in row['trial_type']:
                stim_onset = row['STATE_Fixation1_START']
                stim_offset = row['response_window_end']
                stim_duration = stim_offset - stim_onset
        else: 
            stim_onset = np.nan
            stim_offset = np.nan
            stim_duration= np.nan
    except:
        stim_onset = np.nan
        stim_offset = np.nan
        stim_duration= np.nan
        
    return stim_onset, stim_duration, stim_offset
    


# COMPUTE WINDOW AVERAGE
def compute_window(data, runningwindow):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    for i in range(len(data)):
        if i < runningwindow:
            performance.append(round(np.mean(data[0:i + 1]), 2))
        else:
            performance.append(round(np.mean(data[i - runningwindow:i]), 2))
    return performance


# INSERT ROWS IN A DATAFRAME 
def insert_row(row_number, df, row_value): 
    """
    Inserts a row in a desired position in the slected dataframe
    """
    df1 = df[0:row_number] # Slice the upper half of the dataframe 
    df2 = df[row_number:] # Store the result of lower half of the dataframe 
    df1.loc[row_number]=row_value     # Inser the row in the upper half dataframe 
    df_result = pd.concat([df1, df2])     # Concat the two dataframes 
    df_result.index = [*range(df_result.shape[0])]     # Reassign the index labels 
    return df_result


# CREATE CSVS
def create_csv(df, path):
    df.to_csv(path, sep=';', na_rep='nan', index=False)


# PECRCENTAGE AXES
def axes_pcent(axes, label_kwargs):
    """
    convert y axis form 0-1 to 0-100%
    """
    axes.set_ylabel('Accuracy (%)', label_kwargs)
    axes.set_ylim(0, 1.1)
    axes.set_yticks(np.arange(0, 1.1, 0.1))
    axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])

    
# CHANCE CALCULATION
def chance_calculation(correct_th):
    """
    calculates chance baseline considering the number of possible responses
    """
    screen_size = 1440 * 0.28
    chance = correct_th*2 / screen_size
    return chance


# READ DATAFRAMES 
def read_dataframes(init_time, final_time,subjects_df, events_df):
    """
    returns 4 parsed df: weight_df, water_df, missed_df, task_df
    useful each line is a session with init and final time fefined
    """
    subjects = subjects_df
    subjects['date'] = subjects['date'].astype(str)
    subjects['date'] = pd.to_datetime(subjects['date'], format='%Y/%m/%d %H:%M:%S')
    subjects['basal_weight'] = 1000000
    all_subject_names = sorted(subjects.name.unique())
    for name in all_subject_names:
        try:
            basal = subjects['weight'].loc[(subjects['name'] == name) & (subjects['task'] == 'basal_weight')].iloc[-1]
            subjects.loc[(subjects['name'] == name, 'basal_weight')] = basal
        except:
            pass
    subjects['water'] = pd.to_numeric(subjects['water'])
    subjects['weight'] = pd.to_numeric(subjects['weight'])
    subjects['perc_weight'] = subjects['weight'] / subjects['basal_weight'] * 100

    events = events_df

    events['date'] = events['date'].astype(str)
    events['date'] = pd.to_datetime(events['date'], format='%Y/%m/%d %H:%M:%S')

    subjects = subjects[subjects['date'] > init_time]
    subjects = subjects[subjects['date'] < final_time]

    events = events[events['date'] < final_time]
    events = events[events['date'] > init_time]

    subject_names = sorted(subjects.name.unique())

    weight_df = subjects.loc[subjects['task'] != 'manual_water']
    weight_df = weight_df.drop(columns=['tag', 'water', 'wait_seconds'])
    water_df = subjects.loc[(subjects['task'] != 'control_weight') & (subjects['task'] != 'basal_weight')]
    water_df = water_df.drop(columns=['tag', 'weight', 'basal_weight', 'perc_weight', 'wait_seconds'])

    start_task = events[events['type'] == 'START']
    end_task = events[events['type'] == 'END']
    missed_task = events[events['description'].str.contains('Movement in the|Not allowed to enter until')]

    task_df = pd.DataFrame(columns=['subject', 'start_task', 'end_task', 'task_name', 'stage', 'substage'])
    missed_df = pd.DataFrame(columns=['subject', 'date'])

    try:
        for name in subject_names:
            start_times = start_task['date'].loc[start_task['subject'] == name].tolist()
            task_name_total = start_task['description'].loc[start_task['subject'] == name].tolist()

            try:
                task_list = [task.split('-') for task in task_name_total]
                task_name = [task[0] for task in task_list]
                stage = [int(task[1]) for task in task_list]
                substage = [int(task[2]) for task in task_list]
            except:
                task_name = task_name_total
                stage = [1]*len(task_name_total)
                substage = [1]*len(task_name_total)

            end_times = end_task['date'].loc[end_task['subject'] == name].tolist()
            miss_times = missed_task['date'].loc[missed_task['subject'] == name].tolist()

            start_times2 = []
            end_times2 = []
            task_name2 = []
            stage2 = []
            substage2 = []
            i = 0
            j = 0

            while i < len(start_times) and j < len(end_times):

                if start_times[i] < end_times[j]:
                    if i + 1 < len(start_times):
                        if start_times[i + 1] < end_times[j]:
                            i += 1
                            continue

                    start_times2.append(start_times[i])
                    end_times2.append(end_times[j])
                    task_name2.append(task_name[i])
                    stage2.append(stage[i])
                    substage2.append(substage[i])
                    i += 1
                    j += 1
                else:
                    j += 1

            missed_df2 = pd.DataFrame({'subject': name, 'date': miss_times})

            missed_df = pd.concat([missed_df, missed_df2])


            task_df2 = pd.DataFrame({'subject': name, 'start_task': start_times2, 'end_task': end_times2,
                               'task_name': task_name2, 'stage': stage2, 'substage': substage2})

            task_df = pd.concat([task_df, task_df2])

        weight_df['day'] = weight_df['date'] - timedelta(hours=8)
        weight_df['day'] = weight_df['day'].dt.normalize() + timedelta(hours=20)
        water_df['day'] = water_df['date'] - timedelta(hours=8)
        water_df['day'] = water_df['day'].dt.normalize() + timedelta(hours=20)
        missed_df['day'] = missed_df['date'] - timedelta(hours=8)
        missed_df['day'] = missed_df['day'].dt.normalize() + timedelta(hours=20)
        task_df['day'] = task_df['start_task'] - timedelta(hours=8)
        task_df['day'] = task_df['day'].dt.normalize() + timedelta(hours=20)
        weight_df.rename({'name': 'subject'}, axis=1, inplace=True)
        water_df.rename({'name': 'subject'}, axis=1, inplace=True)
    except:
        pass

    return all_subject_names, weight_df, water_df, missed_df, task_df




# INJECTIONS DAYS CLASSIFICATION
def injection(date, subject, experiment, PBS, RO, MK, SERINE, MUSCI, CNO):
    """Classifies the injected drug by date, creates a column with the treatment"""
        
    if experiment=='Systemic':
        if date == '2021/06/11' and subject == 'MA5':
            return 'Saline'
        elif date == '2021/06/12' and subject == 'MA5':
            return 'Ro63'
        else:
            if date in PBS:
                return 'Saline'
            elif date in RO:
                return 'Ro63'
            elif date in SERINE:
                return 'DSer'
            elif date in MK:
                 return 'MK801'
            else:
                return 'Rest'
            
    elif experiment=='PFC':
        if date == '2021/11/20' and subject == 'A19':
            return 'Saline'
        elif date == '2021/11/20' and subject == 'A20':
            return 'Saline'
        else:
            if date in PBS:
                return 'Saline'
            elif date in RO:
                return 'Ro63'
            elif date in MUSCI:
                return 'M:B'
            elif date in CNO:
                return 'CNO'
            else:
                return 'Rest'

    elif experiment=='ALM':
    
        if date == '2022/01/28' and subject == 'A31': #A31
            return 'Rest'
        elif date == '2022/02/08' and subject == 'A31':
            return 'Rest'

        elif date <= '2022/01/18' and subject == 'A32': #A32
            return 'Rest'
        elif date == '2022/01/19' and subject == 'A32': 
            return 'Saline'
        elif date == '2022/01/27' and subject == 'A32':
            return 'Ro63'
        elif date == '2022/02/05' and subject == 'A32':
            return 'Saline'        
        elif date == '2022/02/06' and subject == 'A32':
            return 'Saline'
        elif date == '2022/02/07' and subject == 'A32':
            return 'Rest'
        elif date == '2022/02/08' and subject == 'A32':
            return 'Saline'
        elif date == '2022/02/09' and subject == 'A32':
            return 'Ro63'

        elif date == '2022/01/20' and subject == 'A33': #A33
            return 'Rest'
        elif date == '2022/01/27' and subject == 'A33':
            return 'Ro63'
        elif date == '2022/02/08' and subject == 'A33':
            return 'M:B'
        elif date == '2022/03/04' and subject == 'A33':
            return 'Saline'

        elif date == '2022/01/27' and subject == 'A35': #A35
            return 'Ro63'

        elif date == '2022/02/08' and subject == 'A37': #A37
            return 'Rest'
        elif date == '2022/03/04' and subject == 'A37':
            return 'Saline'

        elif date <= '2022/01/18' and subject == 'A38': #A38
            return 'Rest'
        elif date == '2022/01/19' and subject == 'A38':
            return 'Saline'
        elif date == '2022/01/27' and subject == 'A38':
            return 'Ro63'       
        elif date == '2022/02/06' and subject == 'A38':
            return 'Saline'
        elif date == '2022/02/07' and subject == 'A38':
            return 'Rest'
        elif date == '2022/02/08' and subject == 'A38':
            return 'Saline'
        elif date == '2022/03/04' and subject == 'A38':
            return 'Saline'

        elif date  <= '2022/02/21' and subject == 'A39': #A39
            return 'Rest'

        elif date < '2022/01/26' and subject == 'A40': #A40
            return 'Rest'
        elif date == '2022/01/28' and subject == 'A40': 
            return 'Rest'
        elif date == '2022/02/07' and subject == 'A40':
            return 'Rest'
        elif date == '2022/02/08' and subject == 'A40':
            return 'Saline'
        elif date == '2022/02/09' and subject == 'A40':
            return 'Ro63'
        elif date == '2022/03/04' and subject == 'A40':
            return 'Saline'

        else:
            if date in PBS:
                return 'Saline'
            elif date in RO:
                return 'Ro63'
            elif date in MUSCI:
                return 'M:B'
            elif date in CNO:
                return 'CNO'
            else:
                return 'Rest'
    else:
        print('ERROR EXPERIMENT!')
      

# SELECTION OF TREATMENT PERIODS
def select(df, period, experiment):
    """Selects the injection period desired for the scpecific experiment"""
    #
    if experiment=='Systemic':
        if period=='Ro':
            selection = df.loc[((df['date']>='2021/05/10') & (df['date']<='2021/07/27'))] #2021/06/16
            colors = ['gray', 'silver', 'orange']
            order= ['Rest', 'Saline', 'Ro63']
        elif period=='MK':
            selection = df.loc[((df['date']>='2021/06/17') & (df['date']<='2021/07/27'))]
            colors = ['gray', 'silver', 'tomato']
            order= ['Rest', 'Saline', 'MK801']
        elif period =='Ser':
            selection = df.loc[((df['date']>='2021/07/28') & (df['date']<='2021/08/06'))]
            colors = ['silver', 'teal']
            order=['Saline', 'Ser']
        elif period =='All':
            selection = df.copy()
            colors= ['gray', 'silver','orange', 'tomato', 'teal']
            order=['Rest', 'Saline', 'Ro63', 'MK801', 'Ser']
        else:
            selection = print('ERROR Systemic!')

    #
    elif experiment=='PFC':
        if period=='Ro':
            selection1 = df.loc[((df['date']>='2021/12/06') & (df['date']<='2021/12/20'))] 
            selection2 = df.loc[((df['date']>='2022/01/02'))] 
            selection = pd.concat([selection1, selection2], ignore_index=True)  
            colors = ['gray', 'silver', 'orange']
            order= ['Rest', 'Saline', 'Ro63']     
        elif period=='M:B':
            selection = df.loc[((df['date']>='2021/11/20')& (df['date']<='2021/12/06'))]
            colors = ['gray', 'silver', 'purple']
            order= ['Rest', 'Saline', 'M:B']
        elif period=='CNO':
            selection = df.loc[((df['date']>='2021/12/14')& (df['date']<='2021/12/31'))]
            selection = selection.loc[((selection['injection']!='Ro63'))]
            colors = ['gray', 'silver', 'yellowgreen']
            order= ['Rest', 'Saline', 'CNO']
        elif period =='All':
            selection = df.loc[((df['date']>='2021/11/21'))]
            colors= ['gray', 'silver', 'purple',  'orange', 'yellowgreen']
            order=['Rest', 'Saline',  'M:B', 'Ro63', 'CNO'] 
        else:
            selection = print('ERROR PFC!')
            
    #
    elif experiment=='ALM':
        if period=='M:B':
            selection = df.loc[((df['date']>='2022/01/10')& (df['date']<='2022/02/25'))]
            selection= selection.loc[selection['injection']!='Ro63']
            selection= selection.loc[selection['subject']!='A39']
            colors = ['gray', 'silver', 'purple']
            order= ['Rest', 'Saline', 'M:B']
        elif period=='Ro':
            selection = df.loc[((df['date']>='2022/01/27') & (df['date']<='2022/03/19'))] 
            selection= selection.loc[selection['injection']!='M:B']
            colors = ['gray', 'silver', 'orange']
            order= ['Rest', 'Saline', 'Ro63']
        elif period=='CNO':
            selection = df.loc[((df['date']>='2022/03/14')& (df['date']<='2022/04/05'))]
            selection = selection.loc[((selection['injection']!='Ro63') & (selection['injection']!='M:B'))]
            selection = selection.loc[~((selection['injection']=='Saline') & (selection['date']<'2022/03/21'))]
            colors = ['gray', 'silver', 'yellowgreen']
            order= ['Rest', 'Saline', 'CNO']
        elif period =='All':
            selection = df.loc[((df['date']>='2022/01/10'))]
            colors= ['gray', 'silver', 'purple',  'orange', 'yellowgreen']
            order=['Rest', 'Saline',  'M:B', 'Ro63', 'CNO']
        else:
            selection = print('ERROR ALM!')
    #
    else:
        selection = print('ERROR EXPERIMENT!')
    return selection, colors, order


# ACCURACY / PERFORMANCE/ MISSES variables
def acc_perf(variable):
    '''Useful to share the same code to plot (acc, perf or misses) changinng only the value var'''
    if variable == 'Performance':
        var='performance'
        hline=0
    elif variable == 'Accuracy':
        var= 'correct_bool'
        hline=chance 
    elif variable == 'Misses':
        var= 'misses'
        hline=0
    return(var, hline)


# Sort animals by its labeling day
def labeling_class(subject):
    ''' Categorize type of labeling for each subject'''
    central_ch = ['A60', 'A61', 'A62', 'A64', 'A65', 'A69','A73', 'A74', 'A75', 'A80',
		  'A53', 'A63', 'A67', 'A68','A76', 'A78', 'A81']

    homecage= [ 'A54', 'A56', 'A66', 'A70', 'A71', 'A79',
		'A55', 'A58', 'A72', 'A77', 'A82']
    if subject in central_ch:
        return 'Central_ch'
    elif subject in homecage:
        return 'Homecage'
    else:
        return 'no labeling'
    
# Statictical marks function
def stats_marks(pvalue):
    ''' convert_pvalue_to_asterisks(pvalue):'''
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


# Genotype
def genotype(subject):
    A=['A83', 'A84', 'A88', 'A89', 'A90', 'A92', 'A94',]
    B=['A85', 'A86', 'A87', 'A91', 'A93', 'A95', 'A96', 'A97']
    if subject in A:
        return('WT')
    elif subject in B:
        return('KO-GRIN1')
    else:
        return np.nan
    
    
def find_last_before(row, x, y, z):
    '''Find the last value in a column X before a value of columnn Y, if not take value column Z'''
    colx_times = row[x]
    coly_time = row[y]
    colz_time = row[z]
    try:
        if colx_times and coly_time:
            filtered_values = [time for time in colx_times if time <= coly_time - 0.2] # tail touches
            if filtered_values:
                last_time_x_before_y = max(filtered_values)
            else:
                last_time_x_before_y = max((time for time in colx_times if time <= coly_time), default=colz_time) 
            return last_time_x_before_y
        else:
            return np.nan  # Return nan if col_times is empty
    except:
        print('colx or y empty')
        return np.nan


def injection_4OHT(date, subject):
    '''Classifies when tamoxifen was injected. 
    Assumes the whole day instead of the specific session'''
    if date == '6/2/23':
        if subject in ['A58', 'A61', 'A62', 'A64', 'A65', 'A66', 'A70']:
            return '4OHT'
        else:
            return 'Rest'
    elif date == '7/3/23':
        if subject in ['A53', 'A54', 'A68', 'A71']:
            return '4OHT'
        else:
            return 'Rest'
    elif date == '20/3/23':
        if subject in ['A56', 'A60', 'A69', 'A72', 'A75', 'A76']:
            return '4OHT'
        else:
            return 'Rest'
    elif date == '27/3/23':
        if subject in ['A58', 'A60', 'A73', 'A74', 'A76', 'A80']:
            return '4OHT'
        else:
            return 'Rest'
    elif date == '10/4/23':
        if subject in ['A55', 'A63', 'A79']:
            return '4OHT'
        else:
            return 'Rest'
    elif date == '19/4/23':
        if subject in ['A78', 'A81', 'A82']:
            return '4OHT'
        else:
            return 'Rest'
    elif date == '27/4/23':
        if subject in ['A57', 'A59']:
            return '4OHT'
        else:
            return 'Rest'
    else:
        return 'Rest'


def immunization(date, subject, nmda_immunized):
    """Classifies the injected drug by date, creates a column with the treatment"""        
    if date == '2024/10/08' or date == '2024/05/11':
        if subject in nmda_immunized:
            return 'Peptide'
        else:
            return 'Saline'    
    elif date == '2024/10/10' or date == '2024/07/11':
            return 'Toxine'
    else:
        return 'Rest'

