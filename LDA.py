#READ DATASET
import pandas as pd
#df=pd.read_csv(r'C:\Users\91813\Downloads\Dataset.csv')
df = pd.read_csv(r'C:\\Users\\DELL\\Downloads\\gpt_dataset_new.csv')




#CHECK FOR MISSING VALUES OR NULL VALUES
print(df['Resume'].isna().sum())
df['Resume'] = df['Resume'].fillna(' ')




#TOKENIZATION
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
def safe_tokenize(df):
    if isinstance(df, str):
        return word_tokenize(df)
    else:
        return []   #Return an empty list if the text is not a string
df['tokens'] = df['Resume'].apply(safe_tokenize)
print(df[['Resume', 'tokens']])




#STOPWORD REMOVAL
from nltk.corpus import stopwords
nltk.download('stopwords')
def remove_stopwords(text):
    tokens = word_tokenize(str(text))
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens
df['filtered_tokens'] = df['Resume'].apply(remove_stopwords)
print(df[['Resume', 'filtered_tokens']])




#LEMMATIZATION
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = word_tokenize(str(text))
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens
                       if word.lower() not in stop_words]
    return filtered_tokens
df['processed_tokens'] = df['Resume'].apply(preprocess_text)
print(df[['Resume', 'processed_tokens']])




#IMPORT GENSIM
from gensim import corpora
from gensim.models import LdaModel
import re




# FUNCTION TO CLEAN TEXT
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)
df['cleaned_resume'] = df['Resume'].apply(clean_text)




# CREATE DICTIONARY AND CORPUS FOR LDA
dictionary = corpora.Dictionary(df['processed_tokens'])
corpus = [dictionary.doc2bow(text) for text in df['processed_tokens']]




# PERFORM LDA 
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)
    



#ASSIGN TOPICS TO EACH DOCUMENT
def get_document_topics(bow):
    return sorted(lda_model.get_document_topics(bow), key=lambda x: -x[1])[:1]
df['dominant_topic'] = df['processed_tokens'].apply(lambda x: get_document_topics(dictionary.doc2bow(x)))




#DISPLAY SAMPLE OF RESULTS
print(df[['Resume', 'dominant_topic']].head())
df['dominant_topic'] = df['dominant_topic'].apply(lambda x: x[0][0] if x else -1)  


# TRAIN A CLASSIFIER USING THE DOMINANT TOPIC AS THE TARGET VARIABLE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(df['processed_tokens'], df['dominant_topic'], test_size=0.2, random_state=42)




# CONVERT PROCESSED TOKENS TO FEATURED VECTORS
def tokens_to_features(tokens):
    return dictionary.doc2bow(tokens)
X_train = [tokens_to_features(tokens) for tokens in X_train]
X_test = [tokens_to_features(tokens) for tokens in X_test]




# CONVERT BOW TO DENSE VECTORS
from gensim.matutils import corpus2dense
X_train_dense = corpus2dense(X_train, num_terms=len(dictionary)).T
X_test_dense = corpus2dense(X_test, num_terms=len(dictionary)).T




# TRAIN USING  RANDOM FOREST CLASSIFIER
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_dense, y_train)
y_pred = clf.predict(X_test_dense)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




# TRAIN USING SVM CLASSIFIER
from sklearn.svm import SVC
svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train_dense, y_train)
y_pred_svm = svm_clf.predict(X_test_dense)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))




# TRAIN USING NAIVE BAYES CLASSIFIER
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
nb_clf = MultinomialNB()
nb_clf.fit(X_train_dense, y_train)
y_pred_nb = nb_clf.predict(X_test_dense)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))




#TRAIN USING KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# Train KNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_dense, y_train)
y_pred_knn = knn_clf.predict(X_test_dense)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))





#TRAIN USING LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_dense, y_train)
y_pred_lr = log_reg.predict(X_test_dense)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))







