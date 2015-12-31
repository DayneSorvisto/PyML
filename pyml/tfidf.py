from collections import Counter
import math
 
#D should be a list of documents (text strings)
#d should be a text string representing an individual document
 
def bag_of_words_dict(d,D):
    list = [w.lower() for w in d.split()]
    return dict(Counter(corpus_dict(list)))
   
def stop_words(s):
    # stop_list can be populated with 'stop words'
    stop_list = ['the','of']
    if s.lower() in stop_list:
        return False
    else:
        return True
 
def corpus_dict(D):
    words = []
    for d in D:
        words.extend(d.split())
    list = filter(stop_words,words)
    list = [ l.lower() for l in list]
    return dict(Counter(list))
 
def tf(word,d,D):
    dict = bag_of_words_dict(d,D)
    return 0.5 + 0.5* dict.get(word,0)/max(dict.values())
 
def idf(word,D):
    dict = corpus_dict(D)
    N = len(dict)
    return math.log(N/dict.get(word))
 
def tfidf(d,D):
     return { word.lower() : (tf(word,d,D) * idf(word,D)) for word in corpus_dict(D)}
