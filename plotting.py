import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
pd.set_option('display.max_columns', None)

sentenceCodes = ['IEO','TIE','IOM','IWW','TAI','MTI','IWL','ITH','DFA','ITS','TSI','WSI']

dictS = {"A": [0, "it's eleven oclock"],
		 "B": [1,"that is exactly what happened"],
         "C": [2,"i'm on my way to the meeting"],
         "D": [3,"i wonder what this is about"],
         "E": [4,"the airplane is almost full"],
         "F": [5,"maybe tomorrow it will be cold"],
         "G": [6,"i would like a new alarm clock"],
         "H": [7,"i think i have a doctor's appointment"],
         "I": [8,"don't forget a jacket"],
         "J": [9,"i think i've seen this before"],
         "K": [10, "the surface is slick"],
         "L": [11,"we'll stop in a couple of minutes"]}

def plot_features(df, feature_list, plot_diffs=False, save_image=False, include_IEO=False):
    
    num_features = len(feature_list)
    
    labels = list(dictS.keys())

    if include_IEO == False:
    	labels = labels[1:]
    
    x = np.arange(len(labels))  # the label locations
    width = .05 * num_features
    
    fig, ax = plt.subplots(figsize=(5*num_features,6))
    width = .7/num_features
    
    offset = -width/2*(num_features-1)
    for i, feature in enumerate(feature_list):
    	if plot_diffs == True:
    		rect = ax.bar(x + offset, list(df['all'] - df[feature]), width, label=feature_list[i])
    	else:
    		rect = ax.bar(x + offset, list(df[feature]), width, label=feature_list[i])
    	offset += width


    if plot_diffs == True:
    	ax.set_ylim(-.1,.1)
    	ax.set_ylabel('WER Difference from Sentence Average')
    	ax.set_title('Group WER Difference from Sentence Average')
    else:
    	ax.set_ylim(0,.6)
    	ax.set_ylabel('WER')
    	ax.set_title('WER By Feature')

    
    ax.set_xlabel('Sentence')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    if save_image == True:
    	if plot_diffs == True:
    		feature_list.append('diffs')
    	if include_IEO == True:
    		feature_list.append('IEO')
    	print(feature_list)
    	plt.savefig('_'.join(feature_list)+'.png')
    plt.show()

def plot_cdf(df, feature, sentence_list=[], save_image=False):
    
    if len(sentence_list) == 0:
        df_test = df
    else:
        df_test = df.loc[df['Sentence'].isin(sentence_list)]
        
    Data = []
    groups = pd.unique(df_test[feature])
    
    for group in groups:
        Data.append(list(df_test.loc[df[feature]==group]['WER']))

    fig, ax = plt.subplots(figsize=(15, 6))    
    
    for i, data in enumerate(Data):
        p = 1. * np.arange(len(data)) / (len(data) - 1)
        ax.plot(p, -np.sort(-np.array(data)), label=groups[i])

    ax.set_ylabel('WER')
    ax.set_xlabel('Proportion')
    ax.set_title('{0} CDF: {1}'.format(feature, ", ".join(sentence_list)))
    ax.legend()
    
    if save_image == True:
        plt.savefig('Distribution_{0}.png'.format(feature))
    plt.show()

if __name__ == "__main__":

	dfAgg = pd.read_pickle('group_sentence_data.pkl')
	dfAgg_noIEO = dfAgg.drop(0)
	dfAgg_noIEO.head(15)

	df = pd.read_pickle('crema-processed-dataset-with-WER-PER.pkl')
	wers = list(df['WER'])
	wers = [float(wer[:-1])*.01 for wer in wers]
	df['WER'] = wers

	# Example plots
	plot_features(dfAgg_noIEO, ['Male', 'Female'], plot_diffs=False)
	plot_features(dfAgg_noIEO, ['Caucasian', 'African American', 'Asian'], plot_diffs=True)

	plot_cdf(df, "Emotion", sentence_list = sentenceCodes[1:])