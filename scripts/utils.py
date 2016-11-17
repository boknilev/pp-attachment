# various utilities
import subprocess
import sys
import operator
from sentence import Sentence, EnglishSentence, ATBSentence, ConllSentence
import numpy as np
import codecs

BW_CHARS = {'\'', '|', '>', '&', '<', '}', 'A', 'b', 'p', 't', 'v', 'j', 'H', 'x', 'd', '*', 'r', 'z', 's', \
            '$', 'D', 'T', 'Z', 'E', 'g', 'f', 'q', 'k', 'l', 'm', 'n','h', 'w', 'y', 'Y', 'F', 'N', 'K', 'a', \
            'i', 'o', '~', 'I', 'O', 'W'}


def increment_dict(dic, key):
    update_dict(dic, key, 1)


def update_dict(dic, key, val):
    if key in dic:
        dic[key] += val
    else:
        dic[key] = val


def update_dict_from_dict(dic, new_dic):
    for key, val in new_dic.iteritems():
        update_dict(dic, key, val)


def read_spmrl_file(spmrl_filename, has_morphs):
    """
    Read an SPMRL file and return list of sentences
    """

    f = open(spmrl_filename)
    lines = f.readlines()
    f.close()
    num_lines = 6 if has_morphs else 5  # number of lines per sentence
    sentences = []
    for i in xrange(len(lines)):
        if i % num_lines == 0:
            tokens = lines[i].strip().split()
            poses = lines[i + 1].strip().split()
            labels = lines[i + 2].strip().split()
            parents = [int(parent) for parent in lines[i + 3].strip().split()]
            if has_morphs:
                morphs = lines[i + 4].strip().split()
                s = Sentence(tokens, poses, labels, parents, morphs)
            else:
                s = Sentence(tokens, poses, labels, parents)
            sentences.append(s)
    return sentences


def read_spmrl_conll_file(spmrl_conll_filename):
    """
    Read a SPMRL .conll file and return list of sentences

    The input file is a SPMRL file converted to conll format by the convert_mst.py script
    Legacy code (try to use read_conll_file instead)
    """

    f = codecs.open(spmrl_conll_filename)
    lines = f.readlines()
    f.close()
    sentences = []
    tokens = []
    lemmas = []
    poses = []
    labels = []
    parents = []
    for line in lines:
        if line.strip() == '':
            s = Sentence(tokens, poses, labels, parents)
            s.set_lemmas(lemmas)
            sentences.append(s)
            tokens = []
            lemmas = []
            poses = []
            labels = []
            parents = []
        else:
            splt = line.strip().split()
            tokens.append(splt[1])
            lemmas.append(splt[2])
            poses.append(splt[4])  # use pos and not cpos
            labels.append(splt[7])
            parents.append(int(splt[6]))
    return sentences


def read_conll_file(conll_filename, language, encoding='utf-8'):
    """
    Read a conll file and return list of sentences

    Input file is a file in conllx format from the conll shared task
    """

    f = codecs.open(conll_filename, encoding=encoding)
    lines = f.readlines()
    f.close()
    sentences = []
    tokens = []
    lemmas = []
    poses = []
    labels = []
    parents = []
    for line in lines:
        if line.strip() == '':
            s = ConllSentence(tokens, poses, labels, parents, lemmas, language)
            sentences.append(s)
            tokens = []
            lemmas = []
            poses = []
            labels = []
            parents = []
        else:
            splt = line.strip().split()
            tokens.append(splt[1])
            lemmas.append(splt[2])
            poses.append(splt[4])  # use pos and not cpos
            labels.append(splt[7])
            parents.append(int(splt[6]))
    return sentences


def read_stanford_atb_conll_file(atb_conll_filename):
    """
    Read a .dep file and return list of sentences

    The input file is an ATB file, prepared by stanford preprocessing scripts, then converted to conll format by the pennconverter tools
    """

    f = codecs.open(atb_conll_filename, encoding='utf-8')
    lines = f.readlines()
    f.close()
    sentences = []
    tokens = []
    poses = []
    labels = []
    parents = []
    for line in lines:
        if line.strip() == '':
            s = ATBSentence(tokens, poses, labels, parents)
            sentences.append(s)
            tokens = []
            poses = []
            labels = []
            parents = []
        else:
            splt = line.strip().split()
            tokens.append(splt[1])
            poses.append(splt[3])
            labels.append(splt[7])
            parents.append(int(splt[6]))
    return sentences


def read_wsj_dep_file(wsj_dep_filename, lower_case=False):
    """
    Read a WSJ .dep file and return list of sentences

    The input file is a WSJ file converted to dependency format by the Penn2Dep converter
    """

    f = open(wsj_dep_filename)
    lines = f.readlines()
    f.close()
    sentences = []
    tokens = []
    poses = []
    labels = []
    parents = []
    start_line = 0
    for i in xrange(len(lines)):
        line = lines[i]
        if line.strip() == '':
            s = EnglishSentence(tokens, poses, labels, parents)
            s.set_start_line(start_line)
            start_line = i+1
            sentences.append(s)
            tokens = []
            poses = []
            labels = []
            parents = []
        else:
            splt = line.strip().split()
            tok = splt[1]
            if lower_case:
                tok = tok.lower()
            tokens.append(tok)
            poses.append(splt[3])
            labels.append(splt[7])
            parents.append(int(splt[6]))
    return sentences


def argmax_two(vals):
    """
    Find indexes of max two values

    This only works when the max value is unique
    """
    best = -1000
    arg_best = -1
    second_best = -1000
    arg_second_best = -1
    for i in xrange(len(vals)):
        if vals[i] > best:
            best = vals[i]
            arg_best = i
    for i in xrange(len(vals)):
        if vals[i] < best and vals[i] > second_best:
            second_best = vals[i]
            arg_second_best = i
    return arg_best, arg_second_best


def bw2utf8(words_bw_str):
    pipe = subprocess.Popen(['perl', 'bw2utf8.pl', words_bw_str], stdout=subprocess.PIPE)
    words_utf8_str = pipe.stdout.read()
    words_utf8_str = words_utf8_str.decode('utf-8')  # fix for str -> unicode conversion
    return words_utf8_str


def get_utf8_map(words_bw):
    """
    Get a map from bw words to their utf-8 transliteration

    words_bw is a set of words
    """

    words_bw_list = list(words_bw)
    words_bw_str = ' '.join(words_bw_list)
    words_utf8_str = bw2utf8(words_bw_str)
    words_utf8_list = words_utf8_str.split()
    m = dict(zip(words_bw_list, words_utf8_list))
    return m


def get_utf8_list(words_bw):
    """
    Get a list of utf-8 words

    words_bw is a set of words or phrases (could be multi words)
    """

    words_bw_list = list(words_bw)
    words_bw_str = '\t'.join(words_bw_list)  # here separator must be '\t' and not space
    words_utf8_str = bw2utf8(words_bw_str)
    words_utf8_list = words_utf8_str.split('\t')
    return words_utf8_list


def get_lemma(el):
    # '6-N-L:HyAp||G:f||N:s||R:I||FG:F||FN:S'
    # '5-PNX-L:)'
    # 11-PNX-L:-
    # 12-N-L:kl||G:m||N:s||R:N||FG:B||FN:B-14
    # 49-P-L:l+-47
    # 11-PNX-L:--0

    #print el
    if 'L:' not in el:
        sys.stderr.write('Error: cannot find lemma in element')
        return
    start = el.find('L:') + 2
    if '||' not in el:
        if el[-1].isdigit():  # 49-P-L:l+-47,    11-PNX-L:--0
            last_hyphen = el.rfind('-')
            end = last_hyphen
        else:  # # '5-PNX-L:)',    # 11-PNX-L:-
            end = len(el)
    else:
        end = el.find('||')
    lemma = el[start:end]
    if 'PIPE' in lemma:
        lemma = lemma.replace('PIPE', '|')
    return lemma


def get_lemma_from_morph(morph):
    # L:mn
    # L:AnAn||G:m||N:s||R:R||FG:M||FN:S
    # L:PIPExr||G:m||N:p||R:N||FG:M||FN:P
    # L:b+

    if 'L:' not in morph:
        sys.stderr.write('Error: cannot find lemma in element')
        return
    start = morph.find('L:') + 2
    if '||' not in morph:
        end = len(morph)
    else:
        end = morph.find('||')
    lemma = morph[start:end]
    if 'PIPE' in lemma:
        lemma = lemma.replace('PIPE', '|')
    return lemma


def get_word_from_tok_el(el):
    """
    Get word from an element of tokens (instead morphs)

    '4-N-AlEmlyp-3'
    '1-V->DAf--1'
    '4-PNX--RRB--2'
    '6-PNX---0'
    """

    return el.split('-', 2)[2].rsplit('-')[0]


def is_legal_bw(st):
    """
    check if string is legal bw transliteration (narrow - only Arabic chars)
    """

    for c in st:
        if c not in BW_CHARS:
            return False
    return True


def combine_parent_prep_child(parent_lemma, parent_det, prep_lemma, child_lemma, child_det, exact_match=False):
    """
    Combine lemmas for search
    """

    term = ''
    if parent_det:
        term += 'Al'
    term += parent_lemma + ' '
    if exact_match:
        term += '"'
    if prep_lemma.endswith('+'):
        if prep_lemma == 'l+' and child_det:
            term += 'll' + child_lemma
        else:
            term += prep_lemma[:-1]
            if child_det:
                term += 'Al'
            term += child_lemma
    else:
        term += prep_lemma + ' '
        if child_det:
            term += 'Al'
        term += child_lemma
    if exact_match:
        term += '"'
    return term


def combine_parent_child(parent_lemma, parent_det, child_lemma, child_det):

    term = ''
    if parent_det:
        term += 'Al'
    term += parent_lemma + ' '
    if child_det:
        term += 'Al'
    term += child_lemma
    return term


def load_word_vectors(word_vectors_filename):
    """
    Load word vectors created by word2vec
    """

    print 'loading word vectors'
    m = dict()
    with open(word_vectors_filename) as f:
        for line in f:
            splt = line.strip().split()
            word = splt[0]
            vector = [float(el) for el in splt[1:]]
            m[word] = vector
    return m


def get_top_words_from_vectors_vec(word_vectors, vec, n):
    word_distances = dict()
    for w in word_vectors:
        dist = np.dot(word_vectors[w], vec)
        word_distances[w] = dist
    sorted_word_distances = sorted(word_distances.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_word_distances[:n]


def get_top_words_from_vectors(word_vectors, word, n):
    """
    Get n words closest to word in vector space
    """

    if word not in word_vectors:
        print 'Warning: word', word, 'not found in word vectors map'
        return []
    vec = word_vectors[word]
    return get_top_words_from_vectors_vec(word_vectors, vec, n)


def cosine_similarity(vec1, vec2):

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_word_vectors(word_vectors_filename):
    """
    Get word vectors from a word2vec generated file
    """

    if 'utf8' in word_vectors_filename:
        return get_word_vectors_utf8(word_vectors_filename)

    word_vectors = dict()
    with open(word_vectors_filename) as f:
        for line in f:
            splt = line.strip().split()
            word = splt[0]
            vector = [float(el) for el in splt[1:]]
            word_vectors[word] = vector
    return word_vectors


def get_word_vectors_utf8(word_vectors_filename, encoding='utf-8'):
    """
    Get word vectors from a word2vec generated file
    """

    word_vectors = dict()
    with codecs.open(word_vectors_filename, encoding=encoding) as f:
        for line in f:
            splt = line.strip().split()
            word = splt[0]
            try:
                vector = [float(el) for el in splt[1:]]
                word_vectors[word] = vector
            except ValueError:
                continue
    return word_vectors


def get_map_from_file(filename):
    """
    Format: first word in each line is the key, followed by all its values
    """

    m = dict()
    for line in open(filename).readlines():
        splt = line.strip().split()
        if len(splt) < 2:
            continue
        m[splt[0]] = splt[1:]
    return m


def get_brown_clusters(filename):

    word_clusters = dict()
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            # e.g. 110 dog 2
            splt = line.strip().split()
            if len(splt) != 3:
                continue
            word_clusters[splt[1]] = splt[0]
    return word_clusters


# F = '/mnt/scratch/belinkov/word2vec-arabic/vectors_arabic_gigaword.txt'
# m = load_word_vectors(F)
# print 'loaded'
# import gc
# gc.collect()

# n = 10
# child = '$rb'
# child_vec = m[child]
# prep = 'fy'
# prep_vec = m[prep]
# parent = 'AfrAT'
# parent_vec = m[parent]
# pred_parent = 'xSwS'
# pred_parent_vec = m[pred_parent]
#
# # print get_top_words_from_vectors(m, child, n)
# # print get_top_words_from_vectors_vec(m, np.add(child_vec, prep_vec), n)
# # print np.dot(child_vec, parent_vec)
# # print np.dot(np.add(child_vec, prep_vec), parent_vec)
# # print np.dot(child_vec, pred_parent_vec)
# # print np.dot(np.add(child_vec, prep_vec), pred_parent_vec)
#
# word1 = 'Emlp'
# word2 = 'bryTAnyA'
# print get_top_words_from_vectors(m, word1, n)
# print get_top_words_from_vectors(m, word2, n)
# print get_top_words_from_vectors_vec(m, np.add(m[word1], m[word2]), n)

