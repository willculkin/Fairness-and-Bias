import pickle
import string
import sys
import numpy
import pandas as pd

dictSentences = {"IEO": "It's eleven o'clock",
                 "TIE": "That is exactly what happened",
                 "IOM": "I'm on my way to the meeting",
                 "IWW": "I wonder what this is about",
                 "TAI": "The airplane is almost full",
                 "MTI": "Maybe tomorrow it will be cold",
                 "IWL": "I would like a new alarm clock",
                 "ITH": "I think I have a doctor's appointment",
                 "DFA": "Don't forget a jacket ",
                 "ITS": "I think I've seen this before",
                 "TSI": "The surface is slick ",
                 "WSI": "We'll stop in a couple of minutes"}

table = str.maketrans('', '', string.punctuation)


def cleanPhrase(old):
    new = old.translate(table)
    new = new.lower()
    return new


# Function to convert
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele + " "

        # return string
    return str1


def compareAbsoluteValues(dict1, dict2, key):
    return abs((dict1.get(key) if dict1.get(key) else 0) - (dict2.get(key) if dict2.get(key) else 0))


def compareValues(dict1, dict2, key):
    return (dict1.get(key) if dict1.get(key) else 0) - (dict2.get(key) if dict2.get(key) else 0)


if __name__ == '__main__':

    with open('crema-processed-dataset-with-WER.p', 'rb') as f:
        data = pickle.load(f)


    countDiffFromTranscript = 0
    wordsNotConsistentWithTranscript = []
    wordsNotConsistentWithOrginial = []
    countDiffFromOrginial = 0
    for number in range(len(data.index)):
        sentenceCode = data.loc[number]["Sentence"]
        S2T = data.loc[number]["S2T-Transcript"]
        S2T = cleanPhrase(S2T)

        S2T = S2T.split()

        for n, i in enumerate(S2T):
            if i == "11":
                S2T[n] = "eleven"
        S2T = listToString(S2T)

        S2T = cleanPhrase(S2T)
        groundTruth = cleanPhrase(dictSentences[sentenceCode])
        groundTruthWordfreq = []
        S2TWordfreq = []

        groundTruth = groundTruth.split()
        S2T = S2T.split()
        for w in groundTruth:
            groundTruthWordfreq.append(groundTruth.count(w))
        for w in S2T:
            S2TWordfreq.append(S2T.count(w))



        pairsOrginial = dict(zip(groundTruth, groundTruthWordfreq))
        pairsTranscript = dict(zip(S2T, S2TWordfreq))



        for key in pairsOrginial:
            tempListOrignoal = []
            if pairsOrginial.get(key) != pairsTranscript.get(key) and  compareValues(pairsOrginial,pairsTranscript , key) > 0:
                countDiffFromOrginial = countDiffFromOrginial + compareAbsoluteValues(pairsOrginial, pairsTranscript,
                                                                                      key)
                tempListOrignoal.append({key : compareValues(pairsOrginial, pairsTranscript, key)})

        wordsNotConsistentWithOrginial.append(tempListOrignoal)


        for key in pairsTranscript:
            tempList = []

            if pairsOrginial.get(key) != pairsTranscript.get(key) and  compareValues(pairsTranscript, pairsOrginial, key) > 0:
                countDiffFromTranscript = countDiffFromTranscript + compareAbsoluteValues(pairsOrginial, pairsTranscript, key)
                tempList.append({key : compareValues(pairsTranscript,pairsOrginial, key)})

        wordsNotConsistentWithTranscript.append(tempList)


        # if WER == '0.00%':
        #     print("ground truth: ", groundTruth)
        #     print("Speech2Text : ", S2T)


    print(len(wordsNotConsistentWithTranscript))
    print(wordsNotConsistentWithOrginial)

    data["InGroundTruthNotS2T"] = wordsNotConsistentWithOrginial
    data["InS2TNotGroundTruth"] = wordsNotConsistentWithTranscript
    pickle.dump(data, open("crema-processed-dataset-updated.p", "wb"))
