import os
from nltk import sent_tokenize
import spacy
import sklearn.feature_extraction.text as sktext
from sklearn import svm
import sklearn.dummy as dummy
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
import pandas as pd
import logging
import random

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.basicConfig(format='%(funcName)s: %(message)s', level=logging.INFO)

nlp = spacy.load('en')
DATA_DIR = "resources/satire"


def load_training_data():
    train_labels_file = "{}/training-class".format(DATA_DIR)
    train_docs = []
    train_dir = "{}/training".format(DATA_DIR)

    with open(train_labels_file) as file_in:
        raw_labels = file_in.readlines()

    for file_label in raw_labels:
        file, label_str = file_label.split()
        label = label_str[0].lower() == 't'  # label_str is 'true' or 'satire'
        fn = os.path.join(train_dir, file)
        with open(fn, encoding="ISO-8859-1") as tfile:
            text = tfile.read()
        train_docs.append({"text": text, "filename": file, "label": label})
    LOGGER.info("Loaded {} documents".format(len(train_docs)))
    return train_docs


def enhance_terms(text, boost_factor):
    doc = nlp(text)
    to_boost = []
    for np in doc.noun_chunks:
        if np.root.dep_ == "nsubj" and np.root.ent_type_ in ["PERSON", "ORG"]:
            to_boost.append(np.text)
            to_boost.append(np.root.head.text)
    text = text + " " + " ".join(to_boost * boost_factor)
    return text.lower()


def enhance_docs(docs, boost_factor):
    LOGGER.info("Applying {}-boost enhancement to {} docs ".format(boost_factor, len(docs)))
    for d in docs:
        d['text'] = enhance_terms(d['text'], boost_factor=boost_factor)
        d['text'] = d['text'].lower()
    return docs


def feature_selection(X_train, y_train, X_test, num_feats):
    ch2 = SelectKBest(chi2, k=num_feats)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    return (X_train, X_test)


def train_test_split(docs, ratio=0.8):
    random.shuffle(docs)
    split_point = int(len(docs) * ratio)
    X_train = [d['text'] for d in docs[0:split_point]]
    X_test = [d['text'] for d in docs[split_point:]]
    y_train = [d['label'] for d in docs[0:split_point]]
    y_test = [d['label'] for d in docs[split_point:]]
    LOGGER.info("Splitting data into {} training, {} testing docs".format(len(X_train), len(X_test)))
    return ([X_train, X_test, y_train, y_test])


def build_model(X_train, X_test, y_train, y_test):
    """Build and evaluate a model. Also returns the test-set predictions."""
    count_vect = sktext.CountVectorizer()
    tfidf_transformer = sktext.TfidfTransformer()

    X_train_counts = count_vect.fit_transform(X_train)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    X_train, X_test = feature_selection(X_train_tfidf, y_train, X_test_tfidf, 'all')

    model = svm.SVC(C=10, kernel='linear')
    # model = dummy.DummyClassifier(strategy="stratified")
    model.fit(X_train, y_train)
    LOGGER.info("Model trained")

    # Test trained model:
    predicted = model.predict(X_test)
    df_out = pd.DataFrame(y_test)
    df_out['pred'] = predicted
    df_out['target'] = y_test
    df_out['match'] = df_out['pred'] == df_out['target']

    classifier = {"model": model, "counter": count_vect, "transformer": tfidf_transformer}
    return (classifier, df_out)


def train_test():
    train_docs = load_training_data()
    train_docs = enhance_docs(train_docs, boost_factor=2)
    X_train, X_test, y_train, y_test = train_test_split(train_docs)
    model, df_out = build_model(X_train, X_test, y_train, y_test)
    rep = metrics.classification_report(y_test, df_out['pred'])
    print(rep)


if __name__ == "__main__":
    random.seed(0)
    train_test()