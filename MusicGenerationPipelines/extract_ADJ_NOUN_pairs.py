
import nltk
from collections import Counter

def get_most_common_adj_noun_pairs(text, n):
    tokens = nltk.word_tokenize(text)
    # Part-of-speech tagging
    tagged = nltk.pos_tag(tokens)
    
    # Find adjective-noun pairs
    adj_noun_pairs = []
    for i in range(len(tagged) - 1):
        if tagged[i][1] in ['JJ'] and tagged[i + 1][1] in ['NN', 'NNS']:
            adj_noun_pairs.append((tagged[i][0], tagged[i + 1][0]))

    # Count the frequency of each adjective-noun pair
    freq = Counter(adj_noun_pairs)
    
    # Remove pairs where any word is smaller than 3 characters
    freq = {pair: count for pair, count in freq.items() if len(pair[0]) > 2 and len(pair[1]) > 2 and pair[0] != "have"}
    
    # Order freq by count
    freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
    print(freq)
    
    # Get the n most frequent pairs
    most_common_pairs = list(freq.keys())[:n]
    most_common_pairs_str = ""
    for t1, t2 in most_common_pairs:
        most_common_pairs_str += t1 + " " + t2 + ", "

    if most_common_pairs_str.endswith(", "):
        most_common_pairs_str = most_common_pairs_str[:-2]

    return most_common_pairs_str


