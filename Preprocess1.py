import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()


class Preprocess:
    def __int__(self):
        pass
      
    def remove_Tags(self,text):
        """
        take string input and clean string without tags.
        use regex to remove the html tags.
        """
        cleaned_text = re.sub('<[^<]+?>', '', text)
        return cleaned_text

    def sentence_tokenize(self,text):
        """
        take string input and return list of sentences.
        use nltk.sent_tokenize() to split the sentences.
        """
        sent_list = []
        for w in nltk.sent_tokenize(text):
            sent_list.append(w)
        return sent_list

    def word_tokenize(self,text):
        """
        :param text:
        :return: list of words
        """
        return [w for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]

    def remove_stopwords(self,sentence):
        """
        removes all the stop words like "is,the,a, etc."
        """
        stop_words = stopwords.words('english')
        return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

   
    def lemmatize(self,text):
        lemmatized_word = [lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        return " ".join(lemmatized_word)

    def get_wordnet_pos(self,word):
    #Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.VERB)

    def stripPunc(self,wordList):
    
        puncList = [";",":","!","?","/","\\",",","#","@","$","&",")","(","\"","'","’","``"]
        for punc in puncList:
            for word in wordList:
                wordList=[word.replace(punc,'') for word in wordList]
        return wordList

    def preprocess(self,text):
        text1 = re.sub(r'\[[0-9]*\]',' ', text) # to remove [0-9] from text
        #text1 = re.sub(r'[^a-zA-Z]',' ', text1)
        sentence_tokens = self.sentence_tokenize(text1) #sentence tokenization
        word_list = []
        for each_sent in sentence_tokens:
            lemmatizzed_sent = self.lemmatize(each_sent) #lemmatization using wordnet lemmatizer with pos
            clean_text = self.remove_Tags(lemmatizzed_sent) #to remove html tags 
            #clean_text = self.remove_stopwords(clean_text) #removes stopwords
            word_tokens = self.word_tokenize(clean_text) #work tokenization
            word_tokens = self.stripPunc(word_tokens) #removes punctuation(";",":","!","?", etc)
            for i in word_tokens:
                word_list.append(i)
        return word_list
    
    
    
