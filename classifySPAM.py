'''
  This program shell reads email data for the spam classification problem.
  The input to the program is the path to the Email directory "corpus" and a limit number.
  The program reads the first limit number of ham emails and the first limit number of spam.
  It creates an "emaildocs" variable with a list of emails consisting of a pair
    with the list of tokenized words from the email and the label either spam or ham.
  It prints a few example emails.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySPAM.py  <corpus directory path> <limit number>
'''
# Step 1
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
import re
import numpy as np
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize, pos_tag
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.stem import PorterStemmer
# define a feature definition function here
from sklearn.metrics import confusion_matrix, classification_report, recall_score
import zipfile
from matplotlib import pyplot as plt

# function to read spam and ham files, train and test a classifier 
def processspamham(dirPath, limitStr, outputPath = None):
  # convert the limit argument from a string to an int
  limit = int(limitStr)

  # extract the zip file if it exists
  # check if the directory "corpus" exists, extract zip if not.
  if not os.path.exists(dirPath):
    print(f"Error: The directory {dirPath} does not exist.")
    # check if the zip file exists
    if not os.path.exists(dirPath+'.zip'):
      print(f"Error: The file {dirPath+'.zip'} does not exist.")
      sys.exit(1)
    # extract the zip file
    print(f"Extracting {dirPath+'.zip'}...")
    zipfile.ZipFile(dirPath + '.zip').extractall()
  
  # start lists for spam and ham email texts
  hamtexts = []
  spamtexts = []
  os.chdir(dirPath)
  # process all files in directory that end in .txt up to the limit
  #    assuming that the emails are sufficiently randomized
  for file in os.listdir("./spam"):
    if (file.endswith(".txt")) and (len(spamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./spam/"+file, 'r', encoding="latin-1")
      spamtexts.append (f.read())
      f.close()
  for file in os.listdir("./ham"):
    if (file.endswith(".txt")) and (len(hamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./ham/"+file, 'r', encoding="latin-1")
      hamtexts.append (f.read())
      f.close()
  
  # print number emails read
  print ("Number of spam files:",len(spamtexts))
  print ("Number of ham files:",len(hamtexts))
  NTotal = len(spamtexts) + len(hamtexts)
  #print
  
  # create list of mixed spam and ham email documents as (list of words, label)
  emaildocs = []
  # add all the spam
  for spam in spamtexts:
    tokens = nltk.word_tokenize(spam)
    emaildocs.append((tokens, 'spam'))
  # add all the regular emails
  for ham in hamtexts:
    tokens = nltk.word_tokenize(ham)
    emaildocs.append((tokens, 'ham'))
  
  # randomize the list
  random.shuffle(emaildocs)

  stop_words = set(stopwords.words('english'))
  emaildocsCleaned= []
  for email, label in emaildocs:
      cleaned_email = [re.sub(r'[^a-zA-Z]', '', word) for word in email]  # Remove non-alphabetic characters
      cleaned_email = [word for word in cleaned_email if word and word not in stop_words]  # Remove any stopwords
      cleaned_email = [word.lower() for word in cleaned_email if len(word)>=4]  # Remove words with length>=4
      emaildocsCleaned.append((cleaned_email,label))  # Add cleaned words to the list of sentences

    # print a few token lists
  for email in emaildocsCleaned[:2]:
    print (email)
    print('\n')
  # possibly filter tokens

  # Step 2 Begins here
  # continue as usual to get all words and create word features

  # Plot a frequency chart for the top 50 nouns/verbs
  def plot_fig(word_list,xlab = 'words'):
    tokens, frequencies = zip(*word_list[:50])
    plt.figure(figsize=(16,9))
    plt.bar(tokens, frequencies)
    plt.xticks(rotation=45)
    plt.title(f'Top-50-{xlab} by frequency')
    plt.xlabel(xlab)
    plt.ylabel('Frequency')
    plt.tight_layout()
    # Save the plot as an image
    plt.savefig(f"Top_{xlab}.png")

  # Create word features
  def get_word_features(emails, k=2000, type='words'):
    all_words = []
    for email,label in emails:
      all_words.extend(email)
    word_dist = FreqDist(all_words)
    sorted_word_dist = sorted(word_dist.items(), key=lambda item: item[1], reverse=True)
    top_words = [word for word,freq  in sorted_word_dist[:k]] #select top features
    plot_fig(sorted_word_dist, xlab = type)
    return list(top_words)  # Top 2000 words as features
  
  # change directory to output path if provided
  if outputPath:
    os.chdir(f"../{outputPath}")
  # feature sets from a feature definition function
  word_features = get_word_features(emaildocsCleaned)

  # Feature extraction function
  def extract_features(email_tokens, word_features):
    return {word: (word in email_tokens) for word in word_features}
  
  # apply feature extraction on the cleaned email tokens from the corpus (emaildocsCleaned)
  feature_sets = [(extract_features(email, word_features), label) for email, label in emaildocsCleaned]
  
  print("############################################################")
  print("#                      ★ Model1★                          #")
  print("############################################################")

  # train classifier and show performance in cross-validation
  n = int(NTotal*0.8)
  train_data = feature_sets[:n]
  test_data = feature_sets[n:]

  # Train classifier
  classifier = NaiveBayesClassifier.train(train_data)

  # Get predictions for test data
  predicted_labels = [classifier.classify(features) for features, label in test_data]
  true_labels = [label for features, label in test_data]

  # Define lists for capuirng Accuracy and Recall values
  Accuracy_list = []
  Recall_list = []

  # Evaluate classifier
  Accuracy_list.append(accuracy(classifier, test_data))
  print("\nAccuracy:", accuracy(classifier, test_data))
  # Confusion matrix
  cm = confusion_matrix(true_labels, predicted_labels, labels=["spam", "ham"])
  print(f"Confusion Matrix:\n{cm}")

  # Classification report
  report = classification_report(true_labels, predicted_labels)
  print(f"Classification Report:\n{report}")
  Recall_list.append(recall_score(true_labels, predicted_labels, pos_label="spam"))

  # Informative Features
  print("\n20 most informative features :")
  classifier.show_most_informative_features(20)

  # Step 3
  # POS tagging
  email_pos_tags = [(pos_tag(email),label) for email, label in emaildocsCleaned]

  print("\n")
  print("############################################################")
  print("#                      ★ Model2★                          #")
  print("############################################################")

  nouns_verbs = []
  for email, label in email_pos_tags:
    nouns_verbs.append( ([word for word, pos in email if pos.startswith('NN') or pos.startswith('VB')], label) )

  # feature sets from a feature definition function
  word_features_nouns_verbs = get_word_features(nouns_verbs, type = 'Nouns')

  feature_sets = [(extract_features(email, word_features_nouns_verbs), label) for email, label in emaildocsCleaned]

  # train classifier and show performance in cross-validation
  n = int(NTotal*0.8)
  random.shuffle(feature_sets)
  train_data = feature_sets[:n]
  test_data = feature_sets[n:]

  # Train classifier for the new set of features
  classifier_model2 = NaiveBayesClassifier.train(train_data)

  # Get predictions for test data
  predicted_labels_nouns_verbs = [classifier_model2.classify(features) for features, label in test_data]
  true_labels_nouns_verbs      = [label for features, label in test_data]

  # Evaluate classifier for model 2
  Accuracy_list.append(accuracy(classifier_model2, test_data))
  print("\nAccuracy:", accuracy(classifier_model2, test_data))
  # Confusion matrix
  cm = confusion_matrix(true_labels_nouns_verbs, predicted_labels_nouns_verbs, labels=["spam", "ham"])
  print(f"Confusion Matrix:\n{cm}")

  # Classification report 
  report_nouns_verbs = classification_report(true_labels_nouns_verbs, predicted_labels_nouns_verbs)
  print(f"Classification Report :\n{report_nouns_verbs}")
  Recall_list.append(recall_score(true_labels_nouns_verbs, predicted_labels_nouns_verbs, pos_label="spam"))

  # Informative Features - nouns and Verbs
  print("\n20 most informative features -Nouns/Verbs :")
  classifier_model2.show_most_informative_features(20)

  print("\n")
  print("############################################################")
  print("#                      ★ Model3★                          #")
  print("############################################################")
 
  adj_adverbs = []
  for email, label in email_pos_tags:
    adj_adverbs.append( ([word for word, pos in email if pos in ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS')], label) )

  # feature sets from a feature definition function
  word_features_adj_adverbs = get_word_features(adj_adverbs, type = 'Adjectives&Adverbs')

  feature_sets = [(extract_features(email, word_features_adj_adverbs), label) for email, label in emaildocsCleaned]

  # train classifier and show performance in cross-validation
  n = int(NTotal*0.8)
  random.shuffle(feature_sets)
  train_data = feature_sets[:n]
  test_data = feature_sets[n:]

  # Train classifier for the new set of features
  classifier_model3 = NaiveBayesClassifier.train(train_data)

  # Get predictions for test data
  predicted_labels_adj_adverbs = [classifier_model3.classify(features) for features, label in test_data]
  true_labels_adj_adverbs      = [label for features, label in test_data]

  # Evaluate classifier for model 3
  Accuracy_list.append(accuracy(classifier_model3, test_data)) 
  print("\nAccuracy:", accuracy(classifier_model3, test_data))
  # Confusion matrix
  cm = confusion_matrix(true_labels_adj_adverbs, predicted_labels_adj_adverbs, labels=["spam", "ham"])
  print(f"Confusion Matrix:\n{cm}")

  # Classification report 
  report_adj_adverbs = classification_report(true_labels_adj_adverbs, predicted_labels_adj_adverbs)
  print(f"Classification Report :\n{report_adj_adverbs}")
  Recall_list.append(recall_score(true_labels_adj_adverbs, predicted_labels_adj_adverbs, pos_label="spam"))

  # Informative Features - Adjectives and Adverbs
  print("\n20 most informative features -Adjectives/Adverbs :")
  classifier_model3.show_most_informative_features(20)

  print("\n")
  print("############################################################")
  print("#                      ★ Model4★                          #")
  print("############################################################")

  sia = SentimentIntensityAnalyzer()

  # Calculate sentiment scores using polarity_scores() for each email
  def calculate_sentiment_scores(email_tokens):
      # Join tokens into a single string
      email_text = ' '.join(email_tokens)
      # Get sentiment score for the email
      sentiment_score = sia.polarity_scores(email_text)
      return sentiment_score['compound']
  
    # Feature extraction function
  def extract_features_sentiment(email_tokens, word_features):
    # Extract regular word features (if the word is in the email)
    features = {word: (word in email_tokens) for word in word_features} 
    # Calculate sentiment score
    sentiment_score = calculate_sentiment_scores(email_tokens)  
    # Add sentiment score as a feature
    features['sentiment_score'] = sentiment_score
    
    return features
  
  # apply feature extraction on the cleaned email tokens from the corpus (emaildocsCleaned)
  feature_sets = [(extract_features_sentiment(email, word_features), label) for email, label in emaildocsCleaned]
  
  # train classifier and show performance in cross-validation
  random.shuffle(feature_sets)
  train_data = feature_sets[:n]
  test_data = feature_sets[n:]

  # Train classifier for the new set of features
  classifier_model4 = NaiveBayesClassifier.train(train_data)

  # Get predictions for test data
  predicted_labels_sentiment_score = [classifier_model4.classify(features) for features, label in test_data]
  true_labels_sentiment_score      = [label for features, label in test_data]
  
  # Evaluate classifier for model 3
  Accuracy_list.append(accuracy(classifier_model4, test_data)) 
  print("\nAccuracy:", accuracy(classifier_model4, test_data))
  # Confusion matrix
  cm = confusion_matrix(true_labels_sentiment_score, predicted_labels_sentiment_score, labels=["spam", "ham"])
  print(f"Confusion Matrix:\n{cm}")

  # Classification report 
  report_sentiment_score = classification_report(true_labels_sentiment_score, predicted_labels_sentiment_score)
  print(f"Classification Report :\n{report_sentiment_score}")
  Recall_list.append(recall_score(true_labels_sentiment_score, predicted_labels_sentiment_score, pos_label="spam"))
  # Comparison Report
  comp_report = pd.DataFrame({'Model':['Baseline', 'POS-Nouns/Verbs','POS-Adverbs/Adjectives', 'Sentiment Score'],
                              'Accuracy':Accuracy_list,
                              'Recall':Recall_list})
  print("\nComparing Results")
  print(comp_report)

  # Plot the results in a bar chart
  X_axis = np.arange(len(comp_report['Model'])) 
  fig, ax1 = plt.subplots(figsize=(16, 9))

  bars1 = ax1.bar(X_axis - 0.2, comp_report['Accuracy'], 0.4, label = 'Accuracy', color='b')
  ax1.set_xlabel('Model')
  ax1.set_ylabel('Accuracy', color='b')
  ax1.tick_params(axis='y', labelcolor='b')
  ax1.set_xticks(X_axis)
  ax1.set_xticklabels(comp_report['Model'])

  # Create a second y-axis
  ax2 = ax1.twinx()
  bars2 = ax2.bar(X_axis + 0.2, comp_report['Recall'], 0.4, label = 'Recall', color='r')
  ax2.set_ylabel('Recall', color='r')
  ax2.tick_params(axis='y', labelcolor='r')
  
  plt.title("Comparison of Models - Accuracy and Recall") 
  # Save the plot as an image
  plt.savefig(f"comparison_plot.png")

"""
commandline interface takes a directory name with ham and spam subdirectories
   and a limit to the number of emails read each of ham and spam
It then processes the files and trains a spam detection classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) not in (3,4)):
        print ('usage: python classifySPAM.py <corpus-dir> <limit> [<output-dir>]')
        sys.exit(0)
    processspamham(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
    print("\nProcessing completed.")
        
