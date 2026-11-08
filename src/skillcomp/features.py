"""Text-only features. Fit on supplied training records, never inside predict."""

def vectorizer(kind):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    if kind == "official_word1000":
        return TfidfVectorizer(max_features=1000)
    word = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
    if kind == "word":
        return word
    if kind == "word_char":
        return FeatureUnion([
            ("word", word),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, max_features=40000, sublinear_tf=True)),
        ])
    raise ValueError(kind)

