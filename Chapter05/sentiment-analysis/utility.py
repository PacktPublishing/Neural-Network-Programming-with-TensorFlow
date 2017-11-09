import re
from collections import Counter


def preProcess(text):
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    wordCount = Counter(words)
    trimmedWords = [word for word in words if wordCount[word] > 5]

    return trimmedWords


def lookupTable(words):
    wordCount = Counter(words)
    sortedVocab = sorted(wordCount, key=wordCount.get, reverse=True)
    int2vocab = {i: word for i, word in enumerate(sortedVocab)}
    vocab2int = {word: i for i, word in int2vocab.items()}

    return int2vocab, vocab2int
