# COMP30027 : PROJECT 2
# Language Classification System
# Kara La'Brooy 757553
# Semester 1 2017

import json
import operator
from collections import defaultdict
import timeit
import math
import csv

# A nested defaultdict for storing each language and their library of n-grams
languages = defaultdict(lambda: defaultdict(int))
# The occurences of each n-gram in training data
gram_count = defaultdict(int)
# The frequency of each language in training data
document_count = defaultdict(int)
# The inverse frequencies of each language in training data
inverse_document_count = defaultdict(int)

# Parses a json file as a list of dictionaries [{},{}]
# n.b. keys include: 'lang' (training), 'displayname', 'location', 'text', 'uid'
def preprocess_data(filename):
    data = []
    with open(filename) as fp:
        for line in fp:
            data.append(json.loads(line))
    return data

# Creates n-grams of size n for a piece of text
def make_ngram(n, text):
    ngrams = {}
    for i in range(len(text)-n+1):
        gram = text[i:i+n]
        if gram in ngrams:
            ngrams[gram] += 1
        else:
            ngrams[gram] = 1

    return ngrams

# Occurences of an n-gram are added to the appropriate language library
def add_to_lib(ngrams, instance):
    lang = instance.get('lang')

    document_count[lang] += 1

    for k, v in ngrams.items():
        gram_count[k] += 1
        languages[lang][k] += 1

# Fill the inverse_document_count dictionary
def calcuate_inverse_counts(total_document_count):
    for lang, count in document_count.items():
        # Inverse is the total number of documents - the document count with this language tag
        inverse_document_count[lang] += total_document_count - document_count[lang]

# Classifies one text instance based on Naive Bayes
def naive_bayes(languages, grams, threshold):

    best_score = 0
    label = "unk"
    #scores = defaultdict(float)

    for lang, ngrams in languages.items():
        log_sum = 1;
        for gram, value in grams.items():
            # If we have seen this gram in testing, examine
            if gram in gram_count:
                # Probability of gram occuring in current language
                word_prob = float(ngrams[gram]) / document_count[lang]

                # Probability of gram occuring in another language
                #print("word_prob= ", grams[gram], " / ", document_count[lang])
                inv_word_prob = float(gram_count[gram] - ngrams[gram]) / inverse_document_count[lang]
                #print("inv_word_prob= ", gram_count[gram] - grams[gram], " / ", inverse_document_count[lang])

                # P(current language | gram)
                naive = float(word_prob) / (word_prob + inv_word_prob)

                # Protect underflow...
                # Protect logarithmic function...
                if naive == 0:
                    naive = 0.01
                elif naive == 1:
                    naive = 0.99

                # Sum logs for the text input / ngrams
                log_sum += math.log(1 - naive) - math.log(naive)

        # Normalise probability
        score = float(1.0) / (1 + math.exp(log_sum))
        #scores[lang] = score

        # Updated highest probability
        if score > best_score:
            best_score = score
            label = lang

    # Classify as "unk" if does not meet accuracy threshold
    if best_score < threshold:
        label = "unk"

    #print(scores)
    #print(best_score)
    return label

# Determines the accuracy of the machine - simple ratio of correct_guesses/guesses
def accuracy(labels, predictions):
    total_predicted = 0
    correct = 0
    for label in labels:
        if label.get("lang") == predictions[total_predicted]:
            correct += 1
        total_predicted += 1

    return correct/total_predicted

# Writes a csv file with the formatting required for Kaggle submission
def output_csv(fp, docids, labels):
    f = open(fp, 'wt')
    try:
        fieldnames = ('docid', 'lang')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        headers = dict((n,n) for n in fieldnames)
        writer.writerow(headers)
        i = 0
        for anid in docids:
            writer.writerow({'docid':docids[i], 'lang':labels[i],})
            i += 1
    finally:
        f.close()


start_time = timeit.default_timer()

# < --- PROGRAM CONTROLLER START --- >
# Adjustable variables
n = 5
threshold = 0.95
# Instance counter
instances = 0

print("TRAINING...")

training_data = preprocess_data("train.json")
for instance in training_data:
    text = instance.get("text")
    # Clean text
    text = text.strip(" %$^&*()[]@")
    ngrams = make_ngram(n, text)
    add_to_lib(ngrams, instance)
    instances += 1

print("LIBRARIES CREATED")

calcuate_inverse_counts(instances)

print ("BEGIN LABELLING")

# CLASSIFYING TEST DATA
labels = []
docids = []

test_data = preprocess_data("test.json")
for instance in test_data:
    text = instance.get("text")
    # Clean text
    text = text.strip(" %$^&*()[]@")
    ngrams = make_ngram(n, text)
    labels.append(naive_bayes(languages, ngrams, threshold))

# Create list of formatted document ids - docids
i = 0
for label in labels:
    thisid = "test"+str(format(i, '04'))
    docids.append(thisid)
    i += 1

# Print classification summary to .csv
output_csv("naiveout.csv", docids, labels)

# Print runtime summary to console
print("Accuracy (%): \t", accuracy(test_data, labels)*100)
print("Execution Time (s): \t", timeit.default_timer() - start_time)
print("Training: \t", len(training_data), "\t Test: \t", len(test_data))
print("n: \t", n, "\t threshold: \t", threshold)

# < --- PROGRAM CONTROLLER END --- >
