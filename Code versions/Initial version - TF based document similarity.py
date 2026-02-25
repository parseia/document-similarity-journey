import nltk
import parsivar
import math
from PyPDF2 import PdfReader
from collections import Counter
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
def token_farsi(text):
    normalizer=parsivar.Normalizer()
    tokenizer=parsivar.Tokenizer()
    stop_words=set(open(r"C:\Users\Dear User\Desktop\LLM\TASK SANAD\persian_stopwords.txt",encoding="utf-8").read().splitlines())
    text=normalizer.normalize(text)
    tokens=tokenizer.tokenize_words(text)
    filtered=[w for w in tokens if w not in stop_words and w.isalpha()]
    return filtered
def token_eng(text):
    token_eng =nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in token_eng if w.lower() not in stop_words and w.isalpha()]
    return filtered
def tf(tokens):
    counts = Counter(tokens)
    max_count = max(counts.values()) if counts else 1
    weights = {word: count / max_count for word, count in counts.items()}
    return weights
def cosine_similarity(vec1,vec2):
    all_words = set(vec1.keys()) | set(vec2.keys())
    v1 = [vec1.get(word, 0) for word in all_words] 
    v2 = [vec2.get(word, 0) for word in all_words]

    dot_product = sum(a * b for a, b in zip(v1, v2))
    w1 = math.sqrt(sum(a ** 2 for a in v1))
    w2 = math.sqrt(sum(b ** 2 for b in v2))
    
    if w1 == 0 or w2 == 0:
        return 0.0
    return dot_product / (w1 * w2)
def doc_opener(path):
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
dis = int(input("1.FARSI 2.ENG"))
if dis == 1 :
    stp1 = doc_opener(r"C:\Users\Dear User\Desktop\LLM\TASK SANAD\docf1.pdf")
    tokenwords_farsi_1 =  token_farsi(stp1)
    vector1 = tf(tokenwords_farsi_1)
    stp2 = doc_opener(r"C:\Users\Dear User\Desktop\LLM\TASK SANAD\docf2.pdf")
    tokenwords_farsi_2 =  token_farsi(stp2)
    vector2 = tf(tokenwords_farsi_2)
    stp3=cosine_similarity(vector1,vector2)
    print(stp3)
else:
    stp1 = doc_opener(r"C:\Users\Dear User\Desktop\LLM\TASK SANAD\doc1.pdf")
    tokenwords_english_1 =  token_eng(stp1)
    vector1 = tf(tokenwords_english_1)
    stp2 = doc_opener(r"C:\Users\Dear User\Desktop\LLM\TASK SANAD\doc2.pdf")
    tokenwords_english_2 =  token_eng(stp2)
    vector2 = tf(tokenwords_english_2)
    stp3=cosine_similarity(vector1,vector2)
    print(stp3)