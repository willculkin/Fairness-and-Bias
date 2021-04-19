import pandas as pd
import pickle
import string
import nltk

# pip install pyton-Levenshtein
# pip install jiwer
from jiwer import wer

try:
    arpabet = nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    arpabet = nltk.corpus.cmudict.dict()
    
dictS = {"IEO": "it's eleven o'clock",
         "TIE": "that is exactly what happened",
         "IOM": "i'm on my way to the meeting",
         "IWW": "i wonder what this is about",
         "TAI": "the airplane is almost full",
         "MTI": "maybe tomorrow it will be cold",
         "IWL": "i would like a new alarm clock",
         "ITH": "i think i have a doctor's appointment",
         "DFA": "don't forget a jacket",
         "ITS": "i think i've seen this before",
         "TSI": "the surface is slick",
         "WSI": "we'll stop in a couple of minutes"}

dictP = {'IEO': 'IH1 T S IH0 L EH1 V AH0 N AH0 K L AA1 K',
 		 'TIE': 'DH AE1 T IH1 Z IH0 G Z AE1 K T L IY0 W AH1 T HH AE1 P AH0 N D',
 		 'IOM': 'AY1 M AA1 N M AY1 W EY1 T UW1 DH AH0 M IY1 T IH0 NG',
 		 'IWW': 'AY1 W AH1 N D ER0 W AH1 T DH IH1 S IH1 Z AH0 B AW1 T',
 		 'TAI': 'DH AH0 EH1 R P L EY2 N IH1 Z AO1 L M OW2 S T F UH1 L',
 		 'MTI': 'M EY1 B IY0 T AH0 M AA1 R OW2 IH1 T W IH1 L B IY1 K OW1 L D',
 		 'IWL': 'AY1 W UH1 D L AY1 K AH0 N UW1 AH0 L AA1 R M K L AA1 K',
 		 'ITH': 'AY1 TH IH1 NG K AY1 HH AE1 V AH0 D AA1 K T ER0 Z AH0 P OY1 N T M AH0 N T',
		 'DFA': 'D OW1 N T F ER0 G EH1 T AH0 JH AE1 K AH0 T',
 		 'ITS': 'AY1 TH IH1 NG K AY1 V S IY1 N DH IH1 S B IH0 F AO1 R',
 		 'TSI': 'DH AH0 S ER1 F AH0 S IH1 Z S L IH1 K',
 		 'WSI': 'W IY1 L S T AA1 P IH0 N AH0 K AH1 P AH0 L AH1 V M IH1 N AH0 T S'}

def getPhonemes(sentence, convertList=True):
    sentence = sentence.split(" ")
    sentence_phonemes = []
    
    for word in sentence:
        try:
            word_phonemes = arpabet[word][0]
        except KeyError:
            if convertList == True:
                return ""
            return []
        for phoneme in word_phonemes:
            sentence_phonemes.append(phoneme)
            
    if convertList == True:
        sentence_phonemes = " ".join(sentence_phonemes)
        
    return sentence_phonemes


if __name__ == "__main__":
	data = pd.read_pickle('crema-processed-dataset-with-WER.p')

	data.drop("Unnamed: 0", axis=1,inplace=True)

	data.rename(columns={"S2T-Transcript":"S2T_Transcript","S2T-Confidence":"S2T_Confidence"},inplace=True)

	preds = data['S2T_Transcript'].to_list()
	preds = [pred.strip(' "').lower().replace('11', 'eleven').replace('\\','') for pred in preds]
	data['S2T_Transcript'] = preds

	data = data[data.S2T_Transcript != 'n/a']
	data.head()

	Yp = data['S2T_Transcript'].to_list()
	Yt = data['Sentence'].to_list()

	PER = []
	for yp, yt in zip(Yp, Yt):
	    per = wer(dictP[yt], getPhonemes(yp))
	    if per == "":
	        PER.append(-1)
	    else:
	        PER.append(per)

	data["PER"] = PER
	data.head()

	data.to_pickle('crema-processed-dataset-with-WER-PER.pkl')