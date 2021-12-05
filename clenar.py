import re
import unicodedata
import string
import spacy
nlp = spacy.load('en_core_web_sm')




def clean1(text, accented=True, special=True, punctuation = True):
  if accented:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  if punctuation:
    text = ''.join([c for c in text if c not in string.punctuation])
  if special:
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    text = re.sub(pat, '', text)
  return text.lower()
  
def stopword(text):
  # text = nlp(text)
  stopwords = nlp.Defaults.stop_words
  lst=[]
  for token in text.split():
      if token.lower() not in stopwords:    #checking whether the word is not 
          lst.append(token)                    #present in the stopword list.
  text = ' '.join(lst)
  return text