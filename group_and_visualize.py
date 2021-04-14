import pandas as pd
import numpy as np
import psweep as ps


def convert_str_pct_to_float(df):
    wer_strs = df['WER']
    wers = []

    for wer_str in wer_strs:
        wers.append(float(wer_str[:-1]) / 100.0)

    df['WER'] = wers
    return df

def extract_features(df):
    # sex
    # race
    # ethnicity
    # emotion
    feature_counts = {
        'sex': {},
        'race': {},
        #'ethnicity': {},
        #'emotion': {},
    }

    feature_options = {
        'sex': [None],
        'race': [None],
        #'ethnicity': [None],
        #'emotion': [None],
    }

    for sex in df['Sex']:
        if sex in feature_counts['sex']:
            feature_counts['sex'][sex] += 1
        else:
            feature_counts['sex'][sex] = 1
            feature_options['sex'].append(sex)

    for race in df['Race']:
        if race in feature_counts['race']:
            feature_counts['race'][race] += 1
        else:
            feature_counts['race'][race] = 1
            feature_options['race'].append(race)

    '''
    for ethnicity in df['Ethnicity']:
        if ethnicity in feature_counts['ethnicity']:
            feature_counts['ethnicity'][ethnicity] += 1
        else:
            feature_counts['ethnicity'][ethnicity] = 1
            feature_options['ethnicity'].append(ethnicity)

    for emotion in df['Emotion']:
        if emotion in feature_counts['emotion']:
            feature_counts['emotion'][emotion] += 1
        else:
            feature_counts['emotion'][emotion] = 1
            feature_options['emotion'].append(emotion)
    '''

    return feature_counts, feature_options

def extract_sentences(df):
    sentences = {}

    for sentence in df['Sentence']:
        if sentence not in sentences:
            sentences[sentence] = True

    return list(sentences.keys())

def convert_option_to_col(option):
    option = list(option.values())

    col_name = ''
    for group in option:
        if group is not None:
            col_name += '{0}-'.format(group)

    if col_name[-1] == '-':
        col_name = col_name[:-1]

    return col_name

def generate_group_by_sentence_df(df, sentences, feature_options):
    gbs_df = pd.DataFrame()
    gbs_df['Sentence'] = sentences

    parameter_grid = ps.pgrid(ps.plist(p_name, feature_options[p_name])
                                  for p_name in feature_options)

    group_wers = {'all': []}
    for option in parameter_grid:
        option_list = list(option.values())
        if option_list.count(None) != len(option_list):
            group_wers[convert_option_to_col(option)] = []


    column_map = [p_name[:1].upper()+p_name[1:] for p_name in feature_options]

    for sentence in sentences:
        sub_df = df.loc[df['Sentence'] == sentence]

        cumul_avg = np.average(sub_df['WER'])
        group_wers['all'].append(cumul_avg)

        # calculate for each feature option
        for feature_option in parameter_grid:
            group_df = sub_df.copy()
            option = list(feature_option.values())

            if option.count(None) != len(option):
                col_name = convert_option_to_col(feature_option)
                #print(col_name)

                for i in range(len(option)):
                    col = column_map[i]
                    val = option[i]
                    if val is not None:
                        group_df = group_df.loc[group_df[col] == val]

                if len(group_df['WER']) > 0:
                    group_avg_wer = np.average(group_df['WER'])
                else:
                    #print(col, val)
                    group_avg_wer = -1

                group_wers[col_name].append(group_avg_wer)


    for col_name in group_wers:
        gbs_df[col_name] = group_wers[col_name]

    return gbs_df

def generate_dsitribution_df(df):
    pass


og_df = pd.read_pickle('./crema-processed-dataset-with-WER.p')
og_df = convert_str_pct_to_float(og_df)

feature_counts, feature_options = extract_features(og_df)
sentences = extract_sentences(og_df)

gbs_df = generate_group_by_sentence_df(og_df, sentences, feature_options)
gbs_df.to_pickle('./group_sentence_data.pkl')
distrib_df = generate_dsitribution_df(og_df)
