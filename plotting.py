import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
pd.set_option('display.max_columns', None)

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

def plot_features(df, feature_list, save_image=False, include_IEO=False):
    
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
        rect = ax.bar(x + offset, list(df[feature]), width, label=feature_list[i])
        offset += width

    ax.set_ylim(0,1)
    ax.set_ylabel('WER')
    ax.set_title('WER By Feature')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    if save_image == True:
    	if include_IEO == True:
    		feature_list = feature_list.append('IEO')
    	plt.savefig('_'.join(feature_list)+'.png')
    plt.show()

if __name__ == "__main__":

	df = pd.read_pickle('group_sentence_data.pkl')
	df_noIEO = df.drop(0)
	df_noIEO.head(15)

	# Example plots
	plot_features(df_noIEO, ['Male', 'Female'])
	plot_features(df_noIEO, ['Caucasian', 'African American', 'Asian'])