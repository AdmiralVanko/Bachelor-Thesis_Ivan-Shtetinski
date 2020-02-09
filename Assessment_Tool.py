#imports
import os
import re

import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import webview

import pandas as pd
import numpy as np

from sklearn.feature_extraction import text
from sklearn import svm
from sklearn import decomposition
from sklearn import pipeline
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.manifold import MDS

import nltk
from nltk.corpus import stopwords
import string
from spellchecker import SpellChecker

import eli5.lime  
from bs4 import BeautifulSoup  
from wordcloud import WordCloud 
import matplotlib.pyplot as plt

#constants
MY_TITLE = "Essay Assessment Tool"
MY_FILETYPES=[('all files', '.*'), ('text files', '.txt')]
FIGCOUNTER = 0
tokenizer = nltk.tokenize.TreebankWordTokenizer()
lemma = nltk.WordNetLemmatizer()
spell = SpellChecker()
stop_words = stopwords.words('english')
regex = re.compile(
    '[%s]' % re.escape(string.punctuation.replace('-'.'')))

def tokenizing(text):
    text = tokenizer.tokenize(text)
    return text

def remove_punct(text):
    text = text.replace("/", " or ")
    #text = text.replace("("," ")
    #text = text.replace(")"," ")
    text = regex.sub('', text)
    #text = text.translate(str.maketrans('','', string.punctuation))
    '''text = "".join([word for word in text
        if word not in string.punctuation])'''
    return text

def remove_stopwords(text):
    text = [word for word in text if word not in stop_words]
    return text
    
def lemmatizing(text):
    text = [lemma.lemmatize(word) for word in text]
    return text 

def preprocessing(text):
    if type(text) == pd.Series:
        text = [entry.lower() for entry in text]
        text = [remove_punct(entry) for entry in text]
        text = [tokenizing(entry) for entry in text]
        text = [remove_stopwords(entry) for entry in text]
        text = [lemmatizing(entry) for entry in text]
        text = [" ".join(entry) for entry in text]
    elif type(text) == str:
        text = text.lower()
        text = remove_punct(text)
        text = tokenizing(text)
        text = remove_stopwords(text)
        text = lemmatizing(text)
        text = " ".join(text)
    return text

def train_classifier(train_data: pd.Series, train_labels) -> pipeline.Pipeline:
    vec = text.TfidfVectorizer(lowercase=False, analyzer='word',
        tokenizer=preprocessing, ngram_range=(1, 2))
    svd = decomposition.TruncatedSVD()
    lsa = pipeline.make_pipeline(vec, svd)
    clf = svm.SVC(gamma="scale", probability=True)
    pipe = pipeline.make_pipeline(lsa, clf)
    pipe.fit(train_data, train_labels)
    return pipe

def save_graph(filename):
    if os.path.isfile(filename):
        os.remove(filename)
    plt.savefig(filename)

def generate_pie_chart(sizes, labels):
    global FIGCOUNTER
    plt.figure(FIGCOUNTER)
    FIGCOUNTER+=1
    plt.pie(sizes, explode=(0,0.1), labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis("off")
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.title("Percantage of text that is misspelled:")
    save_graph("images/spelling_pie.png")

def generate_bar_chart(text, objects, label):
    frequency_dict = dict.fromkeys(objects, 0)
    for entry in text:
        if entry in objects:
            frequency_dict[entry] += 1

    d = {d: v for d, v in sorted(frequency_dict.items(), reverse=True, key=lambda item: item[1])}
    global FIGCOUNTER
    plt.figure(FIGCOUNTER)
    FIGCOUNTER+=1
    plt.bar(range(5), (list(d.values())[0:5]), align='center')
    plt.xticks(range(5), (list(d.keys())[0:5]))
    plt.ylabel(label)
    plt.title("Most misspelled words:")
    save_graph("images/spelling_bar.png") 
    
def generate_spelling_graph(text):
    text = text.lower()
    text = remove_punct(text)
    text = tokenizing(text)
    text = remove_stopwords(text)
    misspelled = spell.unknown(text)
    sizes = [len(text)-len(misspelled), len(misspelled)]
    labels = 'correctly spelled words', 'misspelled words'
    generate_pie_chart(sizes, labels)
    generate_bar_chart(text, misspelled, "Occurance")

def generate_wordcloud(text):
    wc = WordCloud(max_words=35).generate(text)
    wc.to_file("images/wordcloud.png")

def generate_similarities_graph():
    if len(GraphicUserInterfaces.uploaded_files) > 1:
        vectorizer = text.CountVectorizer(input='filename')
        dtm = vectorizer.fit_transform(GraphicUserInterfaces.uploaded_files)
        dtm = dtm.toarray()
        dist = (1-cosine_similarity(dtm))
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        pos = mds.fit_transform(dist)
        xs, xy = pos[:,0], pos[:,1]
        global FIGCOUNTER
        plt.figure(FIGCOUNTER)
        FIGCOUNTER+=1
        names =  [os.path.basename(filename).replace('.txt', '')
            for filename in GraphicUserInterfaces.uploaded_files]
        for x, y, name in zip (xs, xy, names):
            plt.scatter(x, y)
            plt.text(x, y, name)
        save_graph('images/similarities_graph.png')

def generate_prediction(text):
    if GraphicUserInterfaces.classifier != None:
        text_explainer = eli5.lime.TextExplainer(random_state=42)
        text_explainer.fit(text, 
            GraphicUserInterfaces.classifier.predict_proba)
        prediction_string = text_explainer.show_prediction(
            top=(15,15),targets=[3]).data

        soup = BeautifulSoup(prediction_string, 'html.parser')
        span_list = []
        for span in soup.find_all("span"):
            span_list.append(span)

        with open("images/index.html", "a") as html_file:
            for item in span_list:
                html_file.write("%s\n" % item)
    else:
        if os.path.isfile("images/index.html"):
            os.remove("images/index.html")

        tk.messagebox.showinfo("Error", "No training file uploaded")

class GraphicUserInterfaces():
    '''
Class GraphicUserInterfaces includes all the used gui's and functions which
creates windows and widgets.
    '''
    uploaded_files = []
    current_file = None
    classifier = None

    def __init__(self):

        self.root = tk.Tk()
        self.root.title(MY_TITLE)
        self.root.geometry("300x200")

        self.menubar = tk.Menu(self.root)
        self.menubar.add_command(label="How to use",
            command=self.instructions)
        self.instructions_menu = tk.Menu(self.menubar, tearoff=0)

        self.top_frame = tk.Frame(self.root).pack(side=tk.TOP)
        self.bottom_frame = tk.Frame(self.root).pack(side=tk.BOTTOM)

        self.welcome_lbl = tk.Label(self.top_frame,
            text = "Welcome to the {}".format(MY_TITLE)).pack(side=tk.TOP)

        self.upload_train_button = tk.Button(self.top_frame,
            text="Upload Training File",
            command=self.upload_train_file).pack(side=tk.TOP)

        self.upload_test_button = tk.Button(self.top_frame,
            text="Upload File for Grading",
            command=self.upload_test_file).pack(side=tk.TOP)

        self.visualise_text_button = tk.Button(self.top_frame,
            text="Dispay Text Visualisations",
            command=self.visualise).pack(side=tk.TOP)

        self.root.config(menu = self.menubar)
        self.root.mainloop()
        return

    def instructions(self):
        tk.messagebox.showinfo("How to use", "Instructions")
        return
    
    def upload_train_file(self, event=None):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), 
            title="Select a file to train the classifier: ", filetypes=MY_FILETYPES)
        file = pd.read_csv(filename, sep='\t')
        train_data = file['Essay']
        train_data = preprocessing(train_data)
        train_label = file['Grade']
        GraphicUserInterfaces.classifier = train_classifier(train_data, train_label)

    def upload_test_file(self, event=None):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(),
            title="Select a file You would like to grade: ", filetypes=MY_FILETYPES)
        GraphicUserInterfaces.uploaded_files.append(filename)
        GraphicUserInterfaces.current_file = filename
        with open (filename, 'rt') as file:
            contents = file.read()

        processed = preprocessing(contents)
        generate_wordcloud(processed)
        generate_spelling_graph(contents)
        generate_similarities_graph()
        with open("images/index.html", "wt") as file:
            file.write(
            """<p>""" + contents + """</p>
            <img src='wordcloud.png' style='width:400px;height:300px;'>
            <img src='spelling_pie.png' style='width:400px;height:300px;'>
            <img src='spelling_bar.png' style='width:400px;height:300px;'>
            <img src='similarities_graph.png' style='width:400px;height:300px;'>""")

        generate_prediction(contents)
        webview.create_window("""Essay Assessment Dashboard
         for file{}""".format(GraphicUserInterfaces.current_file),
             "images/index.html")
        webview.start(http_server=True)
     
    def visualise(self, event=None):
        webview.create_window("""Essay Assessment Dashboard
         for file{}""".format(GraphicUserInterfaces.current_file),
             "images/index.html")
        webview.start(http_server=True)

GUI = GraphicUserInterfaces()