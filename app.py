import networkx as nx
from werkzeug.utils import secure_filename
import string
from flask import Flask, render_template, url_for, request
from extract import SummarizeUrl
from Preprocess1 import Preprocess
from newspaper import Article
import numpy as np
import pandas as pd
import nltk
from nltk.cluster.util import cosine_distance


# import Preprocess1
PS = Preprocess()
# pre = PS.preprocess()

import spacy

nlp = spacy.load('en')
app = Flask(__name__)

# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen

import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
# Other Packages
import os

import time

timestr = time.strftime("%Y%m%d-%H%M%S")

# Initialize App
app = Flask(__name__)
# Configuration For Uploads

# Fetch Text From Url
def get_text(url):
    page = urlopen(url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return fetched_text


@app.route('/')
def index():
    return render_template('index.html')


# """----------------------------TEXT SUMMARIZATION--------------------------"""

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    # start = time.time()

    if request.method == 'POST':
        rawtext = request.form['rawtext']
        slider = int(request.form['quantity'])
        words_in = sum([i.strip(string.punctuation).isalpha() for i in rawtext.split()])


        def read_text(text):
            text1 = re.sub(r'\[[0-9]*\]', ' ', rawtext)
            sentences = []
            for s in sent_tokenize(text1):
                sentences.append(s)
            return sentences

        def sentence_similarity(sent1, sent2, stopwords=None):
            if stopwords is None:
                stopwords = []

            sent1 = [w.lower() for w in sent1]
            sent2 = [w.lower() for w in sent2]

            all_words = list(set(sent1 + sent2))

            vector1 = [0] * len(all_words)
            vector2 = [0] * len(all_words)

            # build the vector for the first sentence
            for w in sent1:
                if w in stopwords:
                    continue
                vector1[all_words.index(w)] += 1

            # build the vector for the second sentence
            for w in sent2:
                if w in stopwords:
                    continue
                vector2[all_words.index(w)] += 1

            return 1 - cosine_distance(vector1, vector2)

        def build_similarity_matrix(sentences, stop_words):
            # Create an empty similarity matrix
            similarity_matrix = np.zeros((len(sentences), len(sentences)))

            for idx1 in range(len(sentences)):
                for idx2 in range(len(sentences)):
                    if idx1 == idx2:  # ignore if both are same sentences
                        continue
                    similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

            return similarity_matrix

        def generate_summary(file_name, top_n):
            nltk.download("stopwords")
            stop_words = stopwords.words('english')


            # Step 1 =>------------------ Read text anc split it----------------------------
            sentences = read_text(rawtext)
            from Preprocess1 import Preprocess
            PS = Preprocess()
            cleaned_data = PS.preprocess(rawtext)
            cleaned_data = " ".join(cleaned_data)
            tokens = nltk.word_tokenize(cleaned_data)
            final = "".join(
                [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
            sentences2 = []
            for s in sent_tokenize(final):
                sentences2.append(s)

                # remove punctuations, numbers and special characters
            clean_sentences = pd.Series(sentences2).str.replace("[^a-zA-Z]", " ")

            # Step 2 =>--------------------------- Generate Similary Martix across sentences--------------------------
            sentence_similarity_martix = build_similarity_matrix(clean_sentences, stop_words)

            # Step 3 =>---------------------------- Rank sentences in similarity martix---------------------------
            sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
            scores = nx.pagerank(sentence_similarity_graph)

            # Step 4 =>---------------------------- Sort the rank and pick top sentences-------------------------
            ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

            summ = [ranked_sentence[i][1] for i in range(top_n)]
            return summ

        final_summary1 = generate_summary(rawtext, slider)
        final_summary=" ".join(final_summary1)

        words_out = sum([i.strip(string.punctuation).isalpha() for i in final_summary.split()])
        input1 = int(words_in)
        output1 = int(words_out)
        per = int(words_out * 100 / words_in)

    return render_template('text.html', ctext=rawtext, final_summary=final_summary, input_words=words_in,
                           output_words=words_out, per=per)


# """---------------------URL TAB---------------------"""
@app.route('/analyze_url1', methods=['GET', 'POST'])
def analyze_url1():
    if request.method == 'POST':
        raw_url = request.form['raw_url1']
        rawtext = get_text(raw_url)
        slider2 = int(request.form['quantity'])
        words_in = sum([i.strip(string.punctuation).isalpha() for i in rawtext.split()])

        def read_text(text):
            text1 = re.sub(r'\[[0-9]*\]', ' ', rawtext)
            sentences = []
            for s in sent_tokenize(text1):
                sentences.append(s)
            return sentences

        def sentence_similarity(sent1, sent2, stopwords=None):
            if stopwords is None:
                stopwords = []

            sent1 = [w.lower() for w in sent1]
            sent2 = [w.lower() for w in sent2]

            all_words = list(set(sent1 + sent2))

            vector1 = [0] * len(all_words)
            vector2 = [0] * len(all_words)

            # build the vector for the first sentence
            for w in sent1:
                if w in stopwords:
                    continue
                vector1[all_words.index(w)] += 1

            # build the vector for the second sentence
            for w in sent2:
                if w in stopwords:
                    continue
                vector2[all_words.index(w)] += 1

            return 1 - cosine_distance(vector1, vector2)

        def build_similarity_matrix(sentences, stop_words):
            # Create an empty similarity matrix
            similarity_matrix = np.zeros((len(sentences), len(sentences)))

            for idx1 in range(len(sentences)):
                for idx2 in range(len(sentences)):
                    if idx1 == idx2:  # ignore if both are same sentences
                        continue
                    similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

            return similarity_matrix

        def generate_summary(file_name, top_n):
            nltk.download("stopwords")
            stop_words = stopwords.words('english')

            # Step 1 - Read text anc split it
            sentences = read_text(rawtext)
            from Preprocess1 import Preprocess
            PS = Preprocess()
            cleaned_data = PS.preprocess(rawtext)
            cleaned_data = " ".join(cleaned_data)
            tokens = nltk.word_tokenize(cleaned_data)
            final = "".join(
                [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
            sentences2 = []
            for s in sent_tokenize(final):
                sentences2.append(s)

                # remove punctuations, numbers and special characters
            clean_sentences = pd.Series(sentences2).str.replace("[^a-zA-Z]", " ")

            # Step 2 - Generate Similary Martix across sentences
            sentence_similarity_martix = build_similarity_matrix(clean_sentences, stop_words)

            # Step 3 - Rank sentences in similarity martix
            sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
            scores = nx.pagerank(sentence_similarity_graph)

            # Step 4 - Sort the rank and pick top sentences
            ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

            summ = [ranked_sentence[i][1] for i in range(top_n)]
            return summ

        final_summary1 = generate_summary(rawtext, slider2)
        final_summary = " ".join(final_summary1)

        words_out = sum([i.strip(string.punctuation).isalpha() for i in final_summary.split()])
        input1 = int(words_in)
        output1 = int(words_out)
        per = int(words_out * 100 / words_in)

    return render_template('url.html', ctext=rawtext, final_summary=final_summary, input_words=words_in,
                           output_words=words_out, per=per)


# """------------------NEWS ARTICLES-------------------------"""
@app.route('/analyze_url', methods=['GET', 'POST'])
def analyze_url():
    if request.method == 'POST':
        raw_url = request.form['raw_url']
        slider3 = int(request.form['quantity'])
        article = Article(raw_url)
        article.download()
        article.parse()
        title = article.title
        aut = article.authors
        article.nlp()
        key = article.keywords
        rawtext = article.text
        rawtext = rawtext.replace("Image copyright", "")

        image = article.top_image
        final_summary1 = SummarizeUrl(raw_url,slider3)
        final_summary = ' '.join(final_summary1)
        words_in = int(sum([i.strip(string.punctuation).isalpha() for i in rawtext.split()]))
        words_out = int(sum([i.strip(string.punctuation).isalpha() for i in final_summary.split()]))
        per = int(words_out * 100 / words_in)

    return render_template('result1.html', final_summary=final_summary, ctext=rawtext, title=title, author=aut, image=image, input_words=words_in, output_words=words_out, per=per,key_words=key)

# """------------------NEWS ARTICLES-------------------------"""
@app.route('/analyze_url3', methods=['GET', 'POST'])
def analyze_url3():
    if request.method == 'POST':
        raw_url = request.form['raw_url']
        #slider3 = int(request.form['quantity'])
        article = Article(raw_url)
        article.download()
        article.parse()
        title = article.title
        aut = article.authors
        article.nlp()
        key = article.keywords
        # Title = article.title
        rawtext = article.text
        rawtext = rawtext.replace("Image copyright", "")

        image = article.top_image
        # rawtext = get_text(raw_url)
        # final_reading_time = readingTime(rawtext)
        final_summary1 = SummarizeUrl(raw_url,int(8))
        final_summary = ' '.join(final_summary1)
        words_in = int(sum([i.strip(string.punctuation).isalpha() for i in rawtext.split()]))
        words_out = int(sum([i.strip(string.punctuation).isalpha() for i in final_summary.split()]))
        per = int(words_out * 100 / words_in)

    return render_template('result1.html', final_summary=final_summary, ctext=rawtext, title=title, author=aut, image=image, input_words=words_in, output_words=words_out, per=per,key_words=key)




# """-------------------------DOCUMENT TAB------------------------"""
@app.route('/uploads', methods=['GET', 'POST'])
def uploads():
    if request.method == 'POST' and 'txt_data' in request.files:
        file = request.files['txt_data']
        # choice = request.form['saveoption']
        filename = secure_filename(file.filename)
        file.save(os.path.join('static/uploadedfiles', filename))

        # Document Redaction Here
        with open(os.path.join('static/uploadedfiles', filename), 'r+') as f:
            rawtext = f.read()
            words_in = sum([i.strip(string.punctuation).isalpha() for i in rawtext.split()])

            slider3 = int(request.form['quantity'])

            def read_text(text):
                text1 = re.sub(r'\[[0-9]*\]', ' ', rawtext)
                sentences = []
                for s in sent_tokenize(text1):
                    sentences.append(s)
                return sentences

            def sentence_similarity(sent1, sent2, stopwords=None):
                if stopwords is None:
                    stopwords = []

                sent1 = [w.lower() for w in sent1]
                sent2 = [w.lower() for w in sent2]

                all_words = list(set(sent1 + sent2))

                vector1 = [0] * len(all_words)
                vector2 = [0] * len(all_words)

                # build the vector for the first sentence
                for w in sent1:
                    if w in stopwords:
                        continue
                    vector1[all_words.index(w)] += 1

                # build the vector for the second sentence
                for w in sent2:
                    if w in stopwords:
                        continue
                    vector2[all_words.index(w)] += 1

                return 1 - cosine_distance(vector1, vector2)

            def build_similarity_matrix(sentences, stop_words):
                # Create an empty similarity matrix
                similarity_matrix = np.zeros((len(sentences), len(sentences)))

                for idx1 in range(len(sentences)):
                    for idx2 in range(len(sentences)):
                        if idx1 == idx2:  # ignore if both are same sentences
                            continue
                        similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2],
                                                                            stop_words)

                return similarity_matrix

            def generate_summary(file_name, top_n):
                nltk.download("stopwords")
                stop_words = stopwords.words('english')

                # Step 1 - Read text anc split it
                sentences = read_text(rawtext)
                from Preprocess1 import Preprocess
                PS = Preprocess()
                cleaned_data = PS.preprocess(rawtext)
                cleaned_data = " ".join(cleaned_data)
                tokens = nltk.word_tokenize(cleaned_data)
                final = "".join(
                    [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
                sentences2 = []
                for s in sent_tokenize(final):
                    sentences2.append(s)

                    # remove punctuations, numbers and special characters
                clean_sentences = pd.Series(sentences2).str.replace("[^a-zA-Z]", " ")

                # Step 2 - Generate Similary Martix across sentences
                sentence_similarity_martix = build_similarity_matrix(clean_sentences, stop_words)

                # Step 3 - Rank sentences in similarity martix
                sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
                scores = nx.pagerank(sentence_similarity_graph)

                # Step 4 - Sort the rank and pick top sentences
                ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

                summ = [ranked_sentence[i][1] for i in range(top_n)]
                return summ

            final_summary1 = generate_summary(rawtext, slider3)
            final_summary = " ".join(final_summary1)

            words_out = sum([i.strip(string.punctuation).isalpha() for i in final_summary.split()])
            input1 = int(words_in)
            output1 = int(words_out)
            per = int(words_out * 100 / words_in)

    return render_template('doc.html', final_summary=final_summary, ctext=rawtext, input_words=words_in,
                           output_words=words_out, per=per)


if __name__ == '__main__':
    app.run(debug=True)
