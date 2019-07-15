""""""

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import string
import re

from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer().tokenize

# Remove stopwords and punctuation
stop = stopwords.words('english') + list(string.punctuation)

tokenizer = RegexpTokenizer(r'\w+')
porter_stemmer = PorterStemmer()
printable = set(string.printable)

WHITESPACE_REGEX = "[ \t\r\f\n]{1,}"

REGEX_DICT = {
    "&[#\da-zA-Z]*;": "",  # Remove stuff like &apos;
    "\d": r"#",  # Replace all digits with #
    WHITESPACE_REGEX: r" "  # replace all spaces and tabs with a single space
}


def regex_clean(txt, reg, replace):
    return re.sub(re.compile(reg), replace, txt)


def multi_regex_clean(txt, regex_dict=REGEX_DICT):
    for match, replace in regex_dict.items():
        txt = regex_clean(txt, match, replace)
    txt = txt.strip(' \t\n\r')
    return " ".join(txt.split())


def digit_clean(txt):
    out = ""
    for ch in txt:
        if ch.isdigit():
            out += "#"
        else:
            out += ch
    return out


def clean_text(txt, return_string=False, lower=False, remove_digits=True, remove_stop=True):
    """Lowercase, tokenize, remove words that are not informative
    such as stopwords, numbers and punctuations. """
    if isinstance(txt, str):
        if lower:
            txt = txt.lower()  # Lower the text.
        # Split into words. Fast toktok tokenizer.
        txt = toktok(txt)
    # Remove stopwords and digits
    txt = [word for word in txt if (word not in stop if remove_stop else True) \
           and (not word.isdigit() if remove_digits else True)]
    if return_string:
        return " ".join(txt)
    else:
        return txt


def printable_text(txt):
    """Remove not printable chars and all new lines and extra whitespaces"""
    txt = txt.replace('\n', ' ').replace('\r', '').replace("\t", " ")
    txt = txt.strip(' \t\n\r')
    # txt = multi_regex_clean(txt, {WHITESPACE_REGEX: r" "})
    txt = "".join([char for char in txt if char in printable])
    return " ".join(txt.split())


def light_clean(txt):
    """Remove all new lines and extra spaces"""
    # return multi_regex_clean(txt, {WHITESPACE_REGEX: r" "})
    txt = txt.replace('\n', ' ').replace('\r', ' ').replace("\t", " ")
    txt = txt.strip(' \t\n\r')
    # txt = multi_regex_clean(txt, {"[ \t\r\f]{1,}": r" "})
    return " ".join(txt.split())


def clean_stem(txt):
    """Clean and stem words"""
    out = []
    for w in tokenizer.tokenize(txt):
        if w not in stopwords.words('english') and w != '':
            try:
                w_stem = porter_stemmer.stem(w)
                out.append(w_stem)
            except IndexError:
                out.append(w)
    return out
