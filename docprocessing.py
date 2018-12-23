# import spacy
# try: nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'textcat', 'parser'])  # load model package "en_core_web_sm"
# except:
#     import os
#     os.system("python -m spacy download en_core_web_sm")
#     nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'textcat', 'parser'])

def doc_to_structure(doc):
    doc = doc.replace("! ", ". ").replace("? ", ". ").replace(": ", ". ").replace("; ", ". ").replace(") ", ". ").replace(" (", ". ")
    doc = doc.split(". ")
    return (doc)

def check_word(word):
    # word not in nlp.Defaults.stop_words and
    return any(char.isalpha() for char in word)