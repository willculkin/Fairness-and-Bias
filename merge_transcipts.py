import pandas as pd

import time

def load_csv(fn):
    df = pd.read_csv(fn)
    return df

def parse_response(fn):
    lines = []
    with open(fn, 'r') as f:
        for l in f:
            lines.append(l)

    try:
        transcript = lines[1].split(':')[1].strip(' \n\t')
        confidence = lines[2].split(':')[1].strip(' \n\t')
    except:
        transcript = "N/A"
        confidence = -1


    return transcript, confidence


def save_df(df):
    df.to_pickle("./crema-processed-dataset.pkl")

def get_new_cols(df):
    results = []
    confidences = []

    for fn in df['FileName']:
        response_fn = "results/{0}-result.txt".format(fn)
        r,c = parse_response(response_fn)
        results.append(r)
        confidences.append(c)

    return results, confidences

def merge_transcipts():
    audio_df = load_csv('processedData.csv')

    audio_df['S2T-Transcript'], audio_df['S2T-Confidence'] = get_new_cols(audio_df)

    save_df(audio_df)

merge_transcipts()
