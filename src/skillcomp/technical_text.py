"""Fixed input-preservation representations; no corpus-dependent token rules."""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer

from skillcomp.features import vectorizer


VARIANTS = ['F0_clean_word', 'F1_raw_word', 'F2_technical_word', 'F3_raw_word_char']
IDENTIFIER = re.compile(r'(?u)\b[^\W\d]\w*(?:[.$][^\W\d]\w*)*\b')
CAMEL = re.compile(r'[A-Z]+(?=[A-Z][a-z]|[^a-z]|$)|[A-Z]?[a-z]+|[0-9]+')


def normalize_raw(text):
    if not isinstance(text, str):
        raise TypeError('Input text must be a string')
    return ' '.join(text.split())


def technical_tokens(text):
    output = []
    for match in IDENTIFIER.finditer(text):
        token = match.group(); candidates = [token.lower()]
        for segment in re.split(r'[.$]', token):
            candidates.append(segment.lower())
            for part in segment.split('_'):
                candidates.append(part.lower()); candidates.extend(piece.lower() for piece in CAMEL.findall(part))
        # Deduplicate within this occurrence, retaining natural term frequency
        # across separate occurrences of the original identifier.
        output.extend(dict.fromkeys(piece for piece in candidates if len(piece) >= 2))
    return output


def preservation_vectorizer(variant):
    if variant in VARIANTS[:2]:
        return vectorizer('word')
    if variant == 'F2_technical_word':
        return TfidfVectorizer(tokenizer=technical_tokens, token_pattern=None, lowercase=False, max_features=30000,
                               ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    if variant == 'F3_raw_word_char':
        return Pipeline([('features', FeatureUnion([
            ('word', TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', max_features=15000, ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ])), ('normalize', Normalizer(norm='l2'))])
    raise ValueError('Unknown input variant: ' + variant)
