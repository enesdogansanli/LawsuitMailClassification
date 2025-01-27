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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle
import pandas as pd
import re
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

DOCUMENTS = {'Ceza_Muhakemesi_Kanunu': (330,'Madde'),
            'Hukuk_Muhakemeleri_Kanunu': (440,'MADDE'),
            'Icra_Iflas_Kanunu':(360,'Madde'),
            'Turk_Borclar_Kanunu':(630,'MADDE'),
            'Turk_Ceza_Kanunu':(340,'Madde'),
            'Turk_Medeni_Kanunu':(998,'Madde'),
            'Turk_Ticaret_Kanunu':(998,'MADDE')}
LABELS = ['Ceza_Muhakemesi_Kanunu','Hukuk_Muhakemeleri_Kanunu','Icra_Iflas_Kanunu','Turk_Borclar_Kanunu','Turk_Ceza_Kanunu','Turk_Medeni_Kanunu','Turk_Ticaret_Kanunu']

nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('turkish')
add_some_words = ['1','2','bir','3','4','5','6','7','8','9','md','–','i','“','”','iki','üç','dört','beş','altı','yedi','sekiz','dokuz','on','a','b','c','d',
                    'ceza','sayılı','karar','hâkim','kanunun','madde','savcı','mahkeme','inci','olarak','suç','olan','diğer','ilişkin','ii','yer','kararı','tarafından',
                    'isteyebilir','içinde','genel','ancak','ek','sonra','göre','karşı','üncü','üçüncü','verilir']
stop_words.extend(add_some_words)

def PlotData():
    '''
    Veri setindeki herbir kanundan kaç adet madde kullanıldığını grafik olarak göstermeye yararyan fonksiyondur.
    '''
    names = ['Ceza M.','Hukuk M.','İcra','Borçlar','Ceza','Medeni','Ticaret']
    value = []
    for i in range(len(LABELS)):
        value.append(DOCUMENTS[LABELS[i]][0])
    
    plt.bar(names,value)
    plt.xlabel('Kanunlar')
    plt.ylabel('Madde Sayıları')
    plt.show()

def setup_docs():
    '''
    Text formatındaki veriyi dosyalardan okuyarak DataFrame yapısında tutulmasını sağlayan fonksiyondur.
    '''
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
    '''
    Verilen text dosyasına küçük harfe dönüşümü uygulayan ve sayıları metin içerisinden çıkaran fonksiyondur.

    Parametre
        text : Text
    '''
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r"[0 - 9]", " ", text)
    return text

def get_tokens(text):
     tokens = word_tokenize(text)
     tokens = [t for t in tokens if not t in stop_words]
     return tokens

def print_frequency_dist(docs):
    '''
    Herbir kanun içerisindeki en çok geçen kelimeleri yazdırmaya yarayan fonksiyondur.

    Parametre
        docs : List
    '''
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
    '''
    Verilen veri setinin karıştırılmasını ve train-test olarak ayrılmasını sağlayan fonksiyondur.

    Parametre 
        docs : List
    '''
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
    '''
    Verilen modelin başarı metrik sonuçlarını yazdıran fonksiyondur.

    Parametre
        title : String
        classifier : Model
        vectorizer : Vectorizer
        X_test : X_test
        y_test : y_test
    '''
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score (y_test,y_pred,average='micro')
    recall = metrics.recall_score(y_test,y_pred,average='macro')
    f1 = metrics.f1_score(y_test,y_pred,average='weighted')

    print("%s\t%f\t%f\t%f \t%f\n" % (title,accuracy, precision, recall, f1))


def train_classifier(docs):
    '''
    Verilen veri ile modellerin en iyi parametrelerinin bulunmasını ve modellerin eğitilmesini sağlayan fonksiyondur.

    Parametre
        docs : List
    '''
    X_train, X_test, y_train, y_test = get_splits(docs)

    vectorizer = CountVectorizer(ngram_range=(1,3),
                                min_df= 3, analyzer = 'word')
    
    dtm = vectorizer.fit_transform(X_train)

    # Naive Bayes
    params_naive = {
    'alpha' : [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0],
    'fit_prior': [True,False]
    }

    naive_clsf = GridSearchCV(
    estimator=MultinomialNB(),
    param_grid=params_naive,
    cv = 5,
    n_jobs=5,
    verbose=1
    )

    naive_clsf.fit(dtm,y_train)

    naive_params = naive_clsf.best_params_

    naive_bayes_classifier = MultinomialNB(alpha=naive_params['alpha'],fit_prior=naive_params['fit_prior']).fit(dtm,y_train)

    evaluate_classifier("Naive Bayes\tTRAIN\t",naive_bayes_classifier,vectorizer,X_train,y_train)
    evaluate_classifier("Naive Bayes\tTEST\t",naive_bayes_classifier,vectorizer,X_test,y_test)

    clf_filename='models/naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename,'wb'))

    vec_filename = 'models/count_vectorizer.pkl'
    pickle.dump(vectorizer,open(vec_filename,'wb'))


    # Decision Tree
    params_decision = {
    'criterion' : ["gini", "entropy", "log_loss"],
    'splitter' : ["best", "random"],
    'min_samples_leaf' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    'max_depth' : [11,12,13,14,15,16,17,18,19,20]
    }

    decision_clsf = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=params_decision,
    cv = 5,
    n_jobs=5,
    verbose=1
    )

    decision_clsf.fit(dtm,y_train)

    decision_params = decision_clsf.best_params_

    decision_tree_classifier = DecisionTreeClassifier(criterion=decision_params['criterion'],splitter=decision_params['splitter'],
                                                        min_samples_leaf=decision_params['min_samples_leaf'],max_depth=decision_params['max_depth']).fit(dtm,y_train)

    evaluate_classifier("Decision Tree\tTRAIN\t",decision_tree_classifier,vectorizer,X_train,y_train)
    evaluate_classifier("Decision Tree\tTEST\t",decision_tree_classifier,vectorizer,X_test,y_test)

    clf_filename='models/decision_tree_classifier.pkl'
    pickle.dump(decision_tree_classifier, open(clf_filename,'wb'))


    # KNN 
    params_knn = {
    'n_neighbors' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    'weights' : ['uniform', 'distance'],
    'leaf_size' : [10,20,30,40,50],
    }

    knn_clsf = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=params_knn,
    cv = 5,
    n_jobs=5,
    verbose=1
    )

    knn_clsf.fit(dtm,y_train)

    knn_params = knn_clsf.best_params_

    knn_classifier = KNeighborsClassifier(n_neighbors=knn_params['n_neighbors'],weights=knn_params['weights'],
                                            leaf_size=['leaf_size']).fit(dtm,y_train)

    evaluate_classifier("KNN Classifier\tTRAIN\t",knn_classifier,vectorizer,X_train,y_train)
    evaluate_classifier("KNN Classifier\tTEST\t",knn_classifier,vectorizer,X_test,y_test)

    clf_filename='models/knn_classifier.pkl'
    pickle.dump(knn_classifier, open(clf_filename,'wb'))


def classify(text):
    '''
    Verilen metni kaydedilmiş modeller üzerinden tahmin eder ve tahmin sonucunu döndürür.

    Parametre
        text : String
    '''
    clf_filename = 'models/naive_bayes_classifier.pkl'
    nb_clf = pickle.load(open(clf_filename,'rb'))

    vec_filename = 'models/count_vectorizer.pkl'
    vectorizer =pickle.load(open(vec_filename,'rb'))

    pred = nb_clf.predict(vectorizer.transform([text]))

    print(pred[0])

def findFalsePredict():
    '''
    Kaydedilen modeller üzerinden tahmin işlemlerleri ve gerçek değerler karşılaştırılarak yanlış tahmin edilen verilerin belirlenmesini sağlar.
    '''
    clf_filename = 'models/naive_bayes_classifier.pkl'
    nb_clf = pickle.load(open(clf_filename,'rb'))

    vec_filename = 'models/count_vectorizer.pkl'
    vectorizer =pickle.load(open(vec_filename,'rb'))

    wronglist = [] # Yanlış tahmin edilen cümleler

    for i in range(len(LABELS)):
        for j in range(1,DOCUMENTS[LABELS[i]][0]):
            if j<10:
                with open('data/{}/00{}.txt'.format(LABELS[i],j),"r",encoding='utf8') as d:
                    oku = d.read()
                    pred = nb_clf.predict(vectorizer.transform([oku]))
                    if pred[0]!=LABELS[i]:
                        wronglist.append(oku)
            elif 9<j<100:
                with open('data/{}/0{}.txt'.format(LABELS[i],j),"r",encoding='utf8') as d:
                    oku = d.read()
                    pred = nb_clf.predict(vectorizer.transform([oku]))
                    if pred[0]!=LABELS[i]:
                        wronglist.append(oku)
            else:
                with open('data/{}/{}.txt'.format(LABELS[i],j),"r",encoding='utf8') as d:
                    oku = d.read()
                    pred = nb_clf.predict(vectorizer.transform([oku]))
                    if pred[0]!=LABELS[i]:
                        wronglist.append(oku)
    
    print("Veri seti icerisinden yanlis tahmin edilen ornek sayisi: {}".format(len(wronglist)))

    return wronglist


if __name__=='__main__':
    docs = setup_docs()

    print_frequency_dist(docs)

    train_classifier(docs)

    wronglist = findFalsePredict()

    PlotData()

    # Verilen bir metnin hangi sınıfa ait olduğunu tahmin etme işlemi !
    # new_doc = " Taşıyıcı, zıya veya hasardan sorumlu olduğu hâllerde, 880 ilâ 882 nci maddelere göre ödenmesi gereken tazminatı ödedikten başka, taşıma ücretini geri verir ve taşıma ile ilgili vergileri, resimleri ve taşıma işi nedeniyle doğan diğer giderleri de karşılar. Ancak, hasar hâlinde, birinci cümle uyarınca yapılacak ödemeler 880 inci maddenin ikinci fıkrasına göre saptanacak bedel ile orantılı olarak belirlenir. Başkaca zararlar karşılanmaz"
    # classify(new_doc)