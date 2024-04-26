import spacy
from number_parser import parse as np_parse

def extract_action_noun_phrases(text, verbs):
    """
    Extract action noun phrases from text
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    phrases = []

    for token in doc:
        if token.lemma_ in verbs:
            phrase = token.text
            for child in token.children:
                if child.dep_ in ["dobj", "prep", "attr"]:
                    phrase += " " + child.text
                    for grandchild in child.children:
                        if grandchild.dep_ in ["pobj", "compound", "amod"]:
                            phrase += " " + grandchild.text
            phrases.append(phrase)
    return phrases

def parse_mixed_ordinal(text):
    """
    Parse mixed ordinal number
    """
    try:
        num = int(''.join(filter(str.isdigit, text)))
        return num
    except ValueError:
        return text

def parse_number_or_ordinal(text):
    """
    Parse number or ordinal number
    """
    parsed = np_parse(text)
    if parsed == text:
        return parse_mixed_ordinal(text)
    else:
        return parsed