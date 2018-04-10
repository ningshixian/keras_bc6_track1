def get_answer():
    """Get an answer."""
    return True

# 去除标点符号,返回小写
def remove_punctuation(sentence):
    sentence = sentence.lower()
    filters = u'!"#$%&*+-/:;<=>?@[\\]^_`{|}~\t\n'
    for char in filters:
        sentence = sentence.replace(char, ' ')
    sentence = sentence.replace('(', '( ')
    sentence = sentence.replace(')', ' )')
    sentence = sentence.replace(', ', ' , ')
    sentence = re.sub(r' \d+(?:\.\d+)?(%)? ', ' NUMBER ', sentence)
    sentence = sentence.replace("   ", " ")
    return sentence