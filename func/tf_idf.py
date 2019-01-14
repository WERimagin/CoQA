from nltk.tokenize import word_tokenize,sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def cos_sim(v1, v2):
    if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0:
        return np.zeros(1)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def tf_idf_vec(paragraph):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(paragraph).toarray()
    return X

def tf_idf(paragraph,question,num_canditate):
    sentences=sent_tokenize(paragraph)
    sentences.append(question)
    vec=tf_idf_vec(sentences)
    cos={i:cos_sim(v,vec[-1]) for i,v in enumerate(vec[:-1])}
    cos=sorted(cos.items(),key=lambda x:-x[1])
    pred_question=" ".join([sentences[c[0]] for c in cos[0:num_canditate]])
    return pred_question,[c[0] for c in cos[0:num_canditate]]
