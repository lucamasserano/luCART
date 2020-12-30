import pandas as pd
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
nlp = spacy.load("en")


def preliminary_preprocessing(papers_df, replace_numbers_with="0"):

    # strip, lowercase, remove newlines and replace numbers to reduce variation (we don't care about their exact value)
    numbers = re.compile('[0-9]+')
    papers_df.loc[:, "abstract"] = papers_df.abstract.apply(lambda text: 
                                            re.sub(numbers, replace_numbers_with, text.strip().replace("\n", " ").lower()))
    return papers_df


def lemmatization(papers_df):
        
    lemmatized_corpus = []
    for abstract in tqdm(papers_df.loc[:, "abstract"].to_list()):
        lemmas = []
        for token in nlp(abstract): # from spacy
            # remove all tokens with length <= 2
            if len(str(token)) > 2:
                # lemmatize
                lemma = token.lemma_
                lemmas.append(lemma)

        lemmatized_corpus.append(lemmas)
    
    papers_df["lemmatized_abstract"] = lemmatized_corpus
    
    return papers_df


def tfidf_features(papers_df):
    
    # preprocessed corpus is expected to be a list of list of tokens/lemmas
    preprocessed_corpus = [" ".join(abstract) for abstract in papers_df.lemmatized_abstract]

    # tfidf
    tfidf_vectorizer = TfidfVectorizer(analyzer="word",
                                       ngram_range=(1,1),
                                       min_df=0.001,
                                       max_df=0.75, 
                                       stop_words="english", 
                                       sublinear_tf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_corpus)

    papers_df = pd.concat([papers_df, 
                           pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())], 
                          axis=1)

    return papers_df