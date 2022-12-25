import os
import random
import string
import nltk
from nltk import word_tokenize
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle
import pandas as pd
import re


# TODO:
# Veri ayırma işlemi(train-test) daha farklı yapılabilir. Biraz ilkel gibi.
# Stopword kısmını araştır.

# Türkçe için stop word kelimelerin çıkarılması gerekmekte.

nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('turkish')
add_some_words = ['1','2','bir','3','4','5','6','7','8','9','md','–','i','“','”','iki','üç','dört','beş','altı','yedi','sekiz','dokuz','on','a','b','c','d',
                    'ceza','sayılı','karar','hâkim','kanunun','madde','savcı','mahkeme','inci','olarak','suç','olan','diğer','ilişkin','ii','yer','kararı','tarafından',
                    'isteyebilir','içinde','genel','ancak','ek','sonra','göre','karşı','üncü','üçüncü','verilir']
stop_words.extend(add_some_words)


DOCUMENTS = {'Ceza_Muhakemesi_Kanunu': (330,'Madde'),
            'Hukuk_Muhakemeleri_Kanunu': (440,'MADDE'),
            'Icra_Iflas_Kanunu':(360,'Madde'),
            'Turk_Borclar_Kanunu':(630,'MADDE'),
            'Turk_Ceza_Kanunu':(340,'Madde'),
            'Turk_Medeni_Kanunu':(998,'Madde'),
            'Turk_Ticaret_Kanunu':(998,'MADDE')}
LABELS = ['Ceza_Muhakemesi_Kanunu','Hukuk_Muhakemeleri_Kanunu','Icra_Iflas_Kanunu','Turk_Borclar_Kanunu','Turk_Ceza_Kanunu','Turk_Medeni_Kanunu','Turk_Ticaret_Kanunu']
BASE_DIR = 'data'

def create_data_set():
    with open('data.txt','w',encoding='utf8') as outfile:
        for label in LABELS:
            dir = '%s/%s' % (BASE_DIR,label)
            for filename in os.listdir(dir):
                fullfilname = '%s/%s' % (dir,filename)
                print(fullfilname)
                with open (fullfilname,'rb') as file:
                    text = file.read().decode(errors='replace').replace('\n', '')
                    outfile.write('%s\t%s\t%s\n' % (label,filename, text))


def setup_docs():
    docs=[]
    mylist1 = [] # Title
    mylist2 = [] # Text

    for i in range(len(LABELS)):
        for j in range(1,DOCUMENTS[LABELS[i]][0]):
            if j<10:
                with open('data/{}/00{}.txt'.format(LABELS[i],j),"r",encoding='utf8') as d:
                    oku = d.read()
                    mylist1.append(LABELS[i])
                    mylist2.append(oku)
            elif 9<j<100:
                with open('data/{}/0{}.txt'.format(LABELS[i],j),"r",encoding='utf8') as d:
                    oku = d.read()
                    mylist1.append(LABELS[i])
                    mylist2.append(oku)
            else:
                with open('data/{}/{}.txt'.format(LABELS[i],j),"r",encoding='utf8') as d:
                    oku = d.read()
                    mylist1.append(LABELS[i])
                    mylist2.append(oku)

    df = pd.DataFrame(list(zip(mylist1, mylist2)),
                columns =['Class', 'Text'])

    for i in range(len(df)):
        doc = (df['Class'][i],df['Text'][i])
        docs.append(doc)

    return docs


def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r"[0 - 9]", " ", text)
    return text


def get_tokens(text):
     tokens = word_tokenize(text)
     tokens = [t for t in tokens if not t in stop_words]
     return tokens

def print_frequency_dist(docs):
    tokens = defaultdict(list)

    for doc in docs:
        doc_label = doc[0]
        doc_text = clean_text(doc[1])

        doc_tokens = get_tokens(doc_text)

        tokens[doc_label].extend(doc_tokens)
    
    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))


def get_splits(docs):
    # Karıştırma işlemi
    random.shuffle(docs)

    X_train = []
    y_train= []

    X_test= []
    y_test= []

    pivot = int(.80*len(docs))

    for i in range(0,pivot):
        X_train.append(docs[i][1])
        y_train.append(docs[i][0])
    
    for i in range(pivot,len(docs)):
        X_test.append(docs[i][1])
        y_test.append(docs[i][0])
    
    return X_train,X_test,y_train, y_test

def evaluate_classifier(title,classifier,vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    precision = metrics.precision_score (y_test,y_pred,average='micro')
    recall = metrics.recall_score(y_test,y_pred,average='micro')
    f1 = metrics.f1_score(y_test,y_pred,average='micro')

    print("%s\t%f\t%f \t%f\n" % (title, precision, recall, f1))


def train_classifier(docs):
    X_train, X_test, y_train, y_test = get_splits(docs)

    vectorizer = CountVectorizer(ngram_range=(1,3),
                                min_df= 3, analyzer = 'word')
    
    dtm = vectorizer.fit_transform(X_train)

    naive_bayes_classifier = MultinomialNB().fit(dtm,y_train)

    evaluate_classifier("Naive Bayes\tTRAIN\t",naive_bayes_classifier,vectorizer,X_train,y_train)
    evaluate_classifier("Naive Bayes\tTEST\t",naive_bayes_classifier,vectorizer,X_test,y_test)

    # Sürekli olarak model eğitimi yapmamak için modellerimizi kaydediyoruz galiba.
    clf_filename='naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename,'wb'))

    vec_filename = 'count_vectorizer.pkl'
    pickle.dump(vectorizer,open(vec_filename,'wb'))

    decision_tree_classifier = DecisionTreeClassifier().fit(dtm,y_train)

    evaluate_classifier("Decision Tree\tTRAIN\t",decision_tree_classifier,vectorizer,X_train,y_train)
    evaluate_classifier("Decision Tree\tTEST\t",decision_tree_classifier,vectorizer,X_test,y_test)

    clf_filename='decision_tree_classifier.pkl'
    pickle.dump(decision_tree_classifier, open(clf_filename,'wb'))


def classify(text):
    clf_filename = 'naive_bayes_classifier.pkl'
    nb_clf = pickle.load(open(clf_filename,'rb'))

    vec_filename = 'count_vectorizer.pkl'
    vectorizer =pickle.load(open(vec_filename,'rb'))

    pred = nb_clf.predict(vectorizer.transform([text]))

    print(pred[0])


if __name__=='__main__':
    # create_data_set()

    docs = setup_docs()

    # print_frequency_dist(docs)

    train_classifier(docs)

    new_doc = " Temsil yetkisi, bir şubenin işleriyle sınırlandırılabilir. Temsil yetkisi, birden çok kişinin birlikte imza atmaları koşuluyla da sınırlandırılabilir. Bu durumda, diğerlerinin katılımı olmaksızın temsilcilerden birinin imza atmış olması, işletme sahibini bağlamaz. Temsil yetkisine ilişkin yukarıdaki sınırlamalar, ticaret siciline tescil edilmedikçe, iyiniyetli üçüncü kişilere karşı hüküm doğurmaz."

    classify(new_doc)