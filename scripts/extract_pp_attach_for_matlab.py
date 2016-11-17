# extract pp attachments from original files directly into files loaded by matlab

from sentence import *
from utils import *


class Attachment(object):
    """Store a PP attachment"""

    heads = []
    label = -1
    pp_words = []
    pp_parents = []  # indices of parents, starting with 1 (0 means root)
    heads_next = []  # words following the head, if exist, else empty string
    heads_pos = []  # part-of-speech tags (V/N) for the heads

    def __init__(self, heads, label, pp_words, pp_parents, heads_next=None, heads_pos=None):
        self.heads = heads
        self.label = label
        self.pp_words = pp_words
        self.pp_parents = pp_parents
        if heads_next:
            self.heads_next = heads_next
        if heads_pos:
            self.heads_pos = heads_pos

    def has_word_vectors(self, word_vectors, ignore_pp_words=False):
        """
        Check if all words in the attachment have word vectors
        """

        for h in self.heads:
            if h not in word_vectors:
                return False
        if ignore_pp_words:
            return True
        else:
            for w in self.pp_words:
                if w not in word_vectors:
                    return False
            return True


class ConllAttachment(object):
    """
    Store a conll attachment

    ideally this and the spmrl attachment above will be sisters, for now they are too different
    """

    # indices from original sentence (0-based)
    orig_prep_id = -1
    orig_heads_ids = []
    sentence_start_line = -1

    def __init__(self, heads, label, prep, child, heads_next=None, heads_pos=None, heads_next_pos=None):
        self.heads = heads
        self.label = label
        self.prep = prep
        self.child = child
        if heads_next:
            self.heads_next = heads_next
        if heads_pos:
            self.heads_pos = heads_pos
        if heads_next_pos:
            self.heads_next_pos = heads_next_pos

    def has_word_vectors(self, word_vectors):
        """
        Check if all words in the attachment have word vectors
        """

        for h in self.heads:
            if h not in word_vectors:
                return False
        if self.prep not in word_vectors:
            return False
        if self.child not in word_vectors:
            return False
        return True

    def set_prep_id(self, prep_id):
        self.orig_prep_id = prep_id

    def set_heads_ids(self, heads_ids):
        self.orig_heads_ids = heads_ids

    def set_sentence_start_line(self, sentence_start_line):
        self.sentence_start_line = sentence_start_line


class EnglishAttachment(ConllAttachment):
    """
    For backwards compatibility
    """
    pass


def get_pp_attachments_from_wsj_sentence(sentence, max_head_distance, max_child_distance, word_vectors=None, get_heads_next=False, \
                                         use_heads_pos=False, use_heads_next_pos=False):
    """
    Get all valid PP attachments from one (English) sentence

    max_distance is the maximum allowed distance of the candidate heads
    Current version only extracts candidate heads, prepositions, and first child of preposition
    """

    attachments = []
    for i in xrange(len(sentence.poses)-1):
        if EnglishSentence.is_prep(sentence.poses[i]):
            children_indices = sentence.get_children(i+1)
            for child_idx in children_indices:
                if sentence.is_valid_attachment(i, child_idx-1, max_head_distance, max_child_distance):
                    gold_prep_parent = sentence.parents[i]
                    heads = []
                    heads_next = []
                    heads_pos = []
                    heads_next_pos = []
                    heads_ids = []
                    prep = sentence.tokens[i]
                    child = sentence.tokens[child_idx-1]
                    gold_head_label = -1
                    start = max(0, i - max_head_distance)
                    for j in xrange(start, i):  # find candidate heads
                        if EnglishSentence.is_noun(sentence.poses[j]) or EnglishSentence.is_verb(sentence.poses[j]):
                            heads.append(sentence.tokens[j])
                            heads_ids.append(j)
                            if use_heads_pos:
                                pos = sentence.poses[j]
                                pos_num = 1 if pos in ENGLISH_POS_TAGS['verb'] else -1  # verb=1, noun=-1
                                heads_pos.append(str(pos_num))
                            if get_heads_next and j < i-1:
                                heads_next.append(sentence.tokens[j+1])
                            if use_heads_next_pos and j < i-1:
                                heads_next_pos.append(sentence.poses[j+1])
                        if gold_prep_parent == j+1:  # if current candidate is the gold head
                            gold_head_label = len(heads)
                    if gold_head_label == -1:
                        sys.stderr.write('Error: could not find the gold head in sentence:\n' + str(sentence))

                    if len(heads) > 1:  # don't take unambiguous cases
                        if get_heads_next:
                            attachment = EnglishAttachment(heads, gold_head_label, prep, child, heads_next=heads_next, heads_pos=heads_pos, heads_next_pos=heads_next_pos)
                        else:
                            attachment = EnglishAttachment(heads, gold_head_label, prep, child, heads_pos=heads_pos, heads_next_pos=heads_next_pos)
                        attachment.set_prep_id(i)
                        attachment.set_heads_ids(heads_ids)
                        attachment.set_sentence_start_line(sentence.start_line)
                        if word_vectors and attachment.has_word_vectors(word_vectors):
                            attachments.append(attachment)
                        # attachments.append(attachment)
    return attachments


def eval_pred_pp_attachments_from_wsj_sentence(gold_sentence, pred_sentence, max_head_distance, max_child_distance, word_vectors=None, get_heads_next=False):
    """
    Get all valid PP attachments from one (English) sentence

    max_distance is the maximum allowed distance of the candidate heads
    Current version only extracts candidate heads, prepositions, and first child of preposition
    """

    num_total = 0
    num_correct = 0
    for i in xrange(len(gold_sentence.poses)-1):
        if EnglishSentence.is_prep(gold_sentence.poses[i]):
            children_indices = gold_sentence.get_children(i+1)
            for child_idx in children_indices:
                if gold_sentence.is_valid_attachment(i, child_idx-1, max_head_distance, max_child_distance):
                    gold_prep_parent = gold_sentence.parents[i]
                    pred_prep_parent = pred_sentence.parents[i]
                    heads = []
                    heads_next = []
                    prep = gold_sentence.tokens[i]
                    child = gold_sentence.tokens[child_idx-1]
                    gold_head_label = -1
                    start = max(0, i - max_head_distance)
                    for j in xrange(start, i):  # find candidate heads
                        if EnglishSentence.is_noun(gold_sentence.poses[j]) or EnglishSentence.is_verb(gold_sentence.poses[j]):
                            heads.append(gold_sentence.tokens[j])
                            if get_heads_next and j < i-1:
                                heads_next.append(gold_sentence.tokens[j+1])
                        if gold_prep_parent == j+1:  # if current candidate is the gold head
                            gold_head_label = len(heads)
                    if gold_head_label == -1:
                        sys.stderr.write('Error: could not find the gold head in sentence:\n' + str(gold_sentence))

                    if len(heads) > 1:  # don't take unambiguous cases
                        if get_heads_next:
                            attachment = EnglishAttachment(heads, gold_head_label, prep, child, heads_next)
                        else:
                            attachment = EnglishAttachment(heads, gold_head_label, prep, child)
                        if word_vectors and attachment.has_word_vectors(word_vectors):
                            num_total += 1
                            if gold_prep_parent == pred_prep_parent:
                                num_correct += 1
    return num_total, num_correct


def get_pp_attachments_from_sentence(sentence, max_distance, max_span, tokens=False, word_vectors=None, get_heads_next=False, get_heads_pos=False):
    """
    Get all valid PP attachments from one sentence, including full PP phrase

    If tokens=True, word forms will be extracted, otherwise lemmas
    max_distance is the maximum allowed distance of the candidate heads
    max_span is the maximum allowed span of the PP
    if get_heads_next=True, will also extract words following the heads
    """

    attachments = []
    for i in xrange(len(sentence.poses) - 1):
        if sentence.poses[i] == POS_PREP:
            # if abs(sentence.parents[i] - i) > max_distance:
            #     continue
            # if sentence.parents[i] - 1 > i:  # skip left arcs
            #     continue
            # parent_pos = sentence.poses[sentence.parents[i] - 1] if sentence.parents[i] > 0 else 'ROOT'
            # if parent_pos != POS_VERB and parent_pos != POS_NOUN:  # only consider noun or verb parents
            #     continue
            # if sentence.poses[i+1] != POS_NOUN:  # only consider prep-noun phrases (ignore e.g. prep-pronoun)
            #     continue
            if sentence.is_valid_attachment(i, max_distance):
                gold_prep_parent = sentence.parents[i]
                pp_words = []
                pp_parents = []
                heads = []
                heads_next = []
                heads_pos = []
                gold_head_label = -1
                start = max(0, i - max_distance)
                for j in xrange(start, i):  # find candidate heads
                    if sentence.poses[j] == POS_NOUN or sentence.poses[j] == POS_VERB:
                        if tokens:
                            heads.append(sentence.tokens[j])
                            if get_heads_next and j < i-1:
                                heads_next.append(sentence.tokens[j+1])
                            if get_heads_pos:
                                heads_pos.append(sentence.poses[j])
                        else:
                            lemma = get_lemma_from_morph(sentence.morphs[j])
                            if lemma[-1] == 'Y':
                                lemma = lemma[:-1] + 'y'
                            heads.append(lemma)
                            if get_heads_next and j <i-1:
                                next_lemma = get_lemma_from_morph(sentence.morphs[j+1])
                                next_lemma = next_lemma.rstrip('+')
                                if next_lemma[-1] == 'Y':
                                    next_lemma = next_lemma[:-1] + 'y'
                                heads_next.append(next_lemma)
                            if get_heads_pos:
                                heads_pos.append(sentence.poses[j])

                        if gold_prep_parent == j+1:  # if current candidate is the gold head
                            gold_head_label = len(heads)
                if gold_head_label == -1:
                    sys.stderr.write('Error: could not find the gold head in sentence:\n' + str(sentence))

                # first word in the PP is the preposition
                if tokens:
                    pp_words.append(sentence.tokens[i])
                else:
                    prep_lemma = get_lemma_from_morph(sentence.morphs[i])
                    prep_lemma = prep_lemma.rstrip('+')
                    if prep_lemma[-1] == 'Y':
                        prep_lemma = prep_lemma[:-1] + 'y'
                    pp_words.append(prep_lemma)
                pp_parents.append(0)  # and prep's parent is set to 0 (it's the root of the PP tree)
                # now find pp subtree
                for j in xrange(i+1, len(sentence.poses)):
                    if sentence.is_reachable(i+1, j+1):
                        if tokens:
                            pp_words.append(sentence.tokens[j])
                        else:
                            lemma = get_lemma_from_morph(sentence.morphs[j])
                            if lemma[-1] == 'Y':
                                lemma = lemma[:-1] + 'y'
                            pp_words.append(lemma)
                        pp_parents.append(sentence.parents[j]-i)  # offset parents so that tree starts with prep

                if len(pp_parents) == len(pp_words):
                    if len(pp_parents) > max_span + 1:
                        sys.stderr.write('Warning: found PP with too large span in sentence:\n' + str(sentence))
                    elif len(pp_parents) <= 1:
                        sys.stderr.write('Error: did not find children for the prep in sentence:\n' + str(sentence))
                    elif len(heads) > 1:  # don't take unambiguous cases
                        if max(pp_parents) <= len(pp_parents): # don't allow words with parents outside of span
                            if get_heads_next:
                                if get_heads_pos:
                                    attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_next, heads_pos)
                                else:
                                    attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_next)
                            else:
                                if get_heads_pos:
                                    attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_pos=heads_pos)
                                else:
                                    attachment = Attachment(heads, gold_head_label, pp_words, pp_parents)
                            if word_vectors:  # if word vectors are given, only add attachments where all words have vectors
                                if attachment.has_word_vectors(word_vectors):
                                    attachments.append(attachment)
                            else:
                                attachments.append(attachment)
                else:
                    sys.stderr.write('Error: Something is wrong with sentence:\n' + str(sentence))
    return attachments


def eval_pred_pp_attachments_from_sentence(gold_sentence, pred_sentence, max_distance, tokens=False, word_vectors=None, get_heads_next=False, get_heads_pos=False):
    """
    Get all valid PP attachments from one sentence, considering only prep child! (contrary to get_pp_attachments_from_sentence())

    If tokens=True, word forms will be extracted, otherwise lemmas
    max_distance is the maximum allowed distance of the candidate heads
    max_span is the maximum allowed span of the PP
    if get_heads_next=True, will also extract words following the heads
    """

    num_total = 0
    num_correct = 0
    for i in xrange(len(gold_sentence.poses) - 1):
        if gold_sentence.poses[i] == POS_PREP:
            # if abs(sentence.parents[i] - i) > max_distance:
            #     continue
            # if sentence.parents[i] - 1 > i:  # skip left arcs
            #     continue
            # parent_pos = sentence.poses[sentence.parents[i] - 1] if sentence.parents[i] > 0 else 'ROOT'
            # if parent_pos != POS_VERB and parent_pos != POS_NOUN:  # only consider noun or verb parents
            #     continue
            # if sentence.poses[i+1] != POS_NOUN:  # only consider prep-noun phrases (ignore e.g. prep-pronoun)
            #     continue
            if gold_sentence.is_valid_attachment(i, max_distance):
                gold_prep_parent = gold_sentence.parents[i]
                pred_prep_parent = pred_sentence.parents[i]
                pp_words = []
                pp_parents = []
                heads = []
                heads_next = []
                heads_pos = []
                gold_head_label = -1
                start = max(0, i - max_distance)
                for j in xrange(start, i):  # find candidate heads
                    if gold_sentence.poses[j] == POS_NOUN or gold_sentence.poses[j] == POS_VERB:
                        if tokens:
                            heads.append(gold_sentence.tokens[j])
                            if get_heads_next and j < i-1:
                                heads_next.append(gold_sentence.tokens[j+1])
                            if get_heads_pos:
                                heads_pos.append(gold_sentence.poses[j])
                        else:
                            lemma = gold_sentence.lemmas[j]
                            if lemma[-1] == 'Y':
                                lemma = lemma[:-1] + 'y'
                            heads.append(lemma)
                            if get_heads_next and j <i-1:
                                next_lemma = gold_sentence.lemmas[j+1]
                                next_lemma = next_lemma.rstrip('+')
                                if next_lemma[-1] == 'Y':
                                    next_lemma = next_lemma[:-1] + 'y'
                                heads_next.append(next_lemma)
                            if get_heads_pos:
                                heads_pos.append(gold_sentence.poses[j])

                        if gold_prep_parent == j+1:  # if current candidate is the gold head
                            gold_head_label = len(heads)
                if gold_head_label == -1:
                    sys.stderr.write('Error: could not find the gold head in sentence:\n' + str(gold_sentence))

                # first word in the PP is the preposition
                if tokens:
                    pp_words.append(gold_sentence.tokens[i])
                else:
                    prep_lemma = gold_sentence.lemmas[i]
                    prep_lemma = prep_lemma.rstrip('+')
                    if prep_lemma[-1] == 'Y':
                        prep_lemma = prep_lemma[:-1] + 'y'
                    pp_words.append(prep_lemma)
                pp_parents.append(0)  # and prep's parent is set to 0 (it's the root of the PP tree)
                # Hack: take only prep's child
                if tokens:
                    pp_words.append(gold_sentence.tokens[i+1])
                else:
                    lemma = gold_sentence.lemmas[i+1]
                    if lemma[-1] == 'Y':
                        lemma = lemma[:-1] + 'y'
                    pp_words.append(lemma)
                pp_parents.append(gold_sentence.parents[i+1]-i)  # offset parents so that tree starts with prep

                if len(heads) > 1:  # don't take unambiguous cases
                    if get_heads_next:
                        if get_heads_pos:
                            attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_next, heads_pos)
                        else:
                            attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_next)
                    else:
                        if get_heads_pos:
                            attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_pos=heads_pos)
                        else:
                            attachment = Attachment(heads, gold_head_label, pp_words, pp_parents)
                    if word_vectors and attachment.has_word_vectors(word_vectors):  # if word vectors are given, only add attachments where all words have vectors
                        num_total += 1
                        if gold_prep_parent == pred_prep_parent:
                            num_correct += 1
    return num_total, num_correct


def eval_pred_pp_attachments_from_stanford_atb_sentence(gold_sentence, pred_sentence, max_distance, word_vectors=None, get_heads_next=False, get_heads_pos=False):
    """
    Get all valid PP attachments from one sentence, considering only prep child! (contrary to get_pp_attachments_from_sentence())

    This version evaluates atb data (preprocessed by stanford preprocessing script, run with rnn parser, converted to conll format with pennconverter)
    This version uses only tokens (no lemma option)
    max_distance is the maximum allowed distance of the candidate heads
    if get_heads_next=True, will also extract words following the heads
    """

    if not type(gold_sentence) == ATBSentence:
        print 'Warning: this method assumes sentence is ATBSentence, but it is instead:', str(type(gold_sentence))

    num_total = 0
    num_correct = 0
    for i in xrange(len(gold_sentence.poses) - 1):
        if ATBSentence.is_prep(gold_sentence.poses[i]):
            if gold_sentence.is_valid_attachment(i, max_distance):
                gold_prep_parent = gold_sentence.parents[i]
                pred_prep_parent = pred_sentence.parents[i]
                pp_words = []
                pp_parents = []
                heads = []
                heads_next = []
                heads_pos = []
                gold_head_label = -1
                start = max(0, i - max_distance)
                for j in xrange(start, i):  # find candidate heads
                    if ATBSentence.is_noun(gold_sentence.poses[j]) or ATBSentence.is_verb(gold_sentence.poses[j]):
                        heads.append(gold_sentence.tokens[j])
                        if get_heads_next and j < i-1:
                            heads_next.append(gold_sentence.tokens[j+1])
                        if get_heads_pos:
                            heads_pos.append(gold_sentence.poses[j])

                        if gold_prep_parent == j+1:  # if current candidate is the gold head
                            gold_head_label = len(heads)
                if gold_head_label == -1:
                    sys.stderr.write('Error: could not find the gold head in sentence:\n' + str(gold_sentence))

                # first word in the PP is the preposition
                pp_words.append(gold_sentence.tokens[i])
                pp_parents.append(0)  # and prep's parent is set to 0 (it's the root of the PP tree)
                # Hack: take only prep's child
                pp_words.append(gold_sentence.tokens[i+1])
                pp_parents.append(gold_sentence.parents[i+1]-i)  # offset parents so that tree starts with prep

                if len(heads) > 1:  # don't take unambiguous cases
                    if get_heads_next:
                        if get_heads_pos:
                            attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_next, heads_pos)
                        else:
                            attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_next)
                    else:
                        if get_heads_pos:
                            attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_pos=heads_pos)
                        else:
                            attachment = Attachment(heads, gold_head_label, pp_words, pp_parents)
                    if word_vectors and attachment.has_word_vectors(word_vectors):  # if word vectors are given, only add attachments where all words have vectors
                        num_total += 1
                        if gold_prep_parent == pred_prep_parent:
                            num_correct += 1
    return num_total, num_correct


def get_pp_attachments_from_sentence_child_grandchild(sentence, max_distance, tokens=False, word_vectors=None, get_heads_next=False):
    """
    Get all valid PP attachments from one sentence, including preposition's child and grandchild

    If tokens=True, word forms will be extracted, otherwise lemmas
    max_distance is the maximum allowed distance of the candidate heads
    if get_heads_next=True, will also extract words following the heads
    """

    attachments = []
    for i in xrange(len(sentence.poses) - 1):
        if sentence.poses[i] == POS_PREP:
            # if abs(sentence.parents[i] - i) > max_distance:
            #     continue
            # if sentence.parents[i] - 1 > i:  # skip left arcs
            #     continue
            # parent_pos = sentence.poses[sentence.parents[i] - 1] if sentence.parents[i] > 0 else 'ROOT'
            # if parent_pos != POS_VERB and parent_pos != POS_NOUN:  # only consider noun or verb parents
            #     continue
            # if sentence.poses[i+1] != POS_NOUN:  # only consider prep-noun phrases (ignore e.g. prep-pronoun)
            #     continue
            if sentence.is_valid_attachment(i, max_distance):
                gold_prep_parent = sentence.parents[i]
                pp_words = []
                pp_parents = []
                prep = ''
                prep_child = ''
                prep_grandchild = ''
                heads = []
                heads_next = []
                gold_head_label = -1
                start = max(0, i - max_distance)
                for j in xrange(start, i):  # find candidate heads
                    if sentence.poses[j] == POS_NOUN or sentence.poses[j] == POS_VERB:
                        if tokens:
                            heads.append(sentence.tokens[j])
                            if get_heads_next and j < i-1:
                                heads_next.append(sentence.tokens[j+1])
                        else:
                            lemma = get_lemma_from_morph(sentence.morphs[j])
                            if lemma[-1] == 'Y':
                                lemma = lemma[:-1] + 'y'
                            heads.append(lemma)
                            if get_heads_next and j <i-1:
                                next_lemma = get_lemma_from_morph(sentence.morphs[j+1])
                                next_lemma = next_lemma.rstrip('+')
                                if next_lemma[-1] == 'Y':
                                    next_lemma = next_lemma[:-1] + 'y'
                                heads_next.append(next_lemma)

                        if gold_prep_parent == j+1:  # if current candidate is the gold head
                            gold_head_label = len(heads)
                if gold_head_label == -1:
                    sys.stderr.write('Error: could not find the gold head in sentence:\n' + str(sentence))

                # first word in the PP is the preposition
                if tokens:
                    prep = sentence.tokens[i]
                else:
                    prep_lemma = get_lemma_from_morph(sentence.morphs[i])
                    prep_lemma = prep_lemma.rstrip('+')
                    if prep_lemma[-1] == 'Y':
                        prep_lemma = prep_lemma[:-1] + 'y'
                    prep = prep_lemma
                # now find pp subtree (only child and grandchild for now)
                if len(sentence.poses) > i+1:  # there's at least one word after the prep
                    if sentence.parents[i+1] == i+1:  # parent of word following prep is the prep
                        if tokens:
                            prep_child = sentence.tokens[i+1]
                        else:
                            child_lemma = get_lemma_from_morph(sentence.morphs[i+1])
                            child_lemma = child_lemma.rstrip('+')
                            if child_lemma[-1] == 'Y':
                                child_lemma = child_lemma[:-1] + 'y'
                            prep_child = child_lemma
                        if len(sentence.poses) > i+2:  # there're at least two words after the prep
                            if sentence.parents[i+2] == i+2:  # parent of word following prep's child is the child
                                if tokens:
                                    prep_grandchild = sentence.tokens[i+2]
                                else:
                                    grandchild_lemma = get_lemma_from_morph(sentence.morphs[i+2])
                                    grandchild_lemma = grandchild_lemma.rstrip('+')
                                    if grandchild_lemma[-1] == 'Y':
                                        grandchild_lemma = grandchild_lemma[:-1] + 'y'
                                    prep_grandchild = grandchild_lemma

                if len(prep_child) == 0:
                    sys.stderr.write('Error: did not find children for the prep in sentence:\n' + str(sentence))
                elif len(heads) > 1:  # don't take unambiguous cases
                    pp_words = [prep, prep_child]
                    pp_parents = [0, 1]
                    if len(prep_grandchild) > 0:
                        pp_words.append(prep_grandchild)
                        pp_parents.append(2)
                    if get_heads_next:
                        attachment = Attachment(heads, gold_head_label, pp_words, pp_parents, heads_next)
                    else:
                        attachment = Attachment(heads, gold_head_label, pp_words, pp_parents)
                    if word_vectors:  # if word vectors are given, only add attachments where all heads have vectors
                        if attachment.has_word_vectors(word_vectors, True):
                            attachments.append(attachment)
                    else:
                        attachments.append(attachment)
    return attachments


def extract_pp_attachments_from_file(spmrl_filename, max_distance, max_span, tokens=False, word_vectors=None, get_heads_next=False, only_child_grandchild=False):
    """
    Get all valid PP attachments from a file

    Return a list of attachments
    """

    print 'extracting attachments from file:', spmrl_filename
    sentences = read_spmrl_file(spmrl_filename, True)
    attachments = []
    for sentence in sentences:
        if only_child_grandchild:
            cur_attachments = get_pp_attachments_from_sentence_child_grandchild(sentence, max_distance, tokens, word_vectors, get_heads_next)
        else:
            cur_attachments = get_pp_attachments_from_sentence(sentence, max_distance, max_span, tokens, word_vectors, get_heads_next)
        attachments += cur_attachments
    return attachments


def eval_pp_attachments_from_pred_file(spmrl_conll_gold_filename, spmrl_conll_pred_filename, max_head_distance, tokens=False, word_vectors=None, use_heads_next=False):

    print 'extracting attachments from file:', spmrl_conll_gold_filename
    gold_sentences = read_spmrl_conll_file(spmrl_conll_gold_filename)
    pred_sentences = read_spmrl_conll_file(spmrl_conll_pred_filename)
    num_total = 0
    num_correct = 0
    if len(gold_sentences) != len(pred_sentences):
        print 'Error: len of gold and pred sentences not equal'
        return
    for i in xrange(len(gold_sentences)):
        cur_total, cur_correct = eval_pred_pp_attachments_from_sentence(gold_sentences[i], pred_sentences[i], max_head_distance,  tokens, word_vectors, use_heads_next)
        num_total += cur_total
        num_correct += cur_correct

    print 'total valid attachments:', num_total, 'correct:', num_correct, '(', "%.2f" %(100.0*num_correct/num_total), '%)'


def eval_pp_attachments_from_stanford_atb_pred_file(atb_conll_gold_filename, atb_conll_pred_filename, max_head_distance, word_vectors=None, use_heads_next=False):

    print 'extracting attachments from file:', atb_conll_gold_filename
    gold_sentences = read_stanford_atb_conll_file(atb_conll_gold_filename)
    pred_sentences = read_stanford_atb_conll_file(atb_conll_pred_filename)
    num_total = 0
    num_correct = 0
    if len(gold_sentences) != len(pred_sentences):
        print 'Error: len of gold and pred sentences not equal'
        return
    for i in xrange(len(gold_sentences)):
        cur_total, cur_correct = eval_pred_pp_attachments_from_stanford_atb_sentence(gold_sentences[i], pred_sentences[i], max_head_distance, word_vectors, use_heads_next)
        num_total += cur_total
        num_correct += cur_correct

    print 'total valid attachments:', num_total, 'correct:', num_correct, '(', "%.2f" %(100.0*num_correct/num_total), '%)'


def extract_pp_attachments_from_wsj_dep_file(wsj_dep_filename, max_head_distance, max_child_distance, word_vectors=None, \
                                             use_heads_next=False, use_heads_pos=False, use_heads_next_pos=False):

    print 'extracting attachments from file:', wsj_dep_filename
    sentences = read_wsj_dep_file(wsj_dep_filename, True)
    attachments = []
    for sentence in sentences:
        assert(type(sentence) == EnglishSentence)
        cur_attachments = get_pp_attachments_from_wsj_sentence(sentence, max_head_distance, max_child_distance, word_vectors, \
                                                               use_heads_next, use_heads_pos, use_heads_next_pos)
        attachments += cur_attachments
    return attachments


def eval_pp_attachments_from_wsj_pred_dep_file(wsj_gold_dep_filename, wsj_pred_dep_filename, max_head_distance, max_child_distance, word_vectors=None, use_heads_next=False):

    print 'extracting attachments from file:', wsj_gold_dep_filename
    gold_sentences = read_wsj_dep_file(wsj_gold_dep_filename, True)
    pred_sentences = read_wsj_dep_file(wsj_pred_dep_filename, True)
    num_total = 0
    num_correct = 0
    if len(gold_sentences) != len(pred_sentences):
        print 'Error: len of gold and pred sentences not equal'
        return
    for i in xrange(len(gold_sentences)):
        cur_total, cur_correct = eval_pred_pp_attachments_from_wsj_sentence(gold_sentences[i], pred_sentences[i], max_head_distance, max_child_distance, word_vectors, use_heads_next)
        num_total += cur_total
        num_correct += cur_correct

    print 'total valid attachments:', num_total, 'correct:', num_correct, '(', "%.2f" %(100.0*num_correct/num_total), '%)'


def eval_pred_pp_attachments_from_conll_sentence(gold_sentence, pred_sentence, max_head_distance, max_child_distance, use_tokens=True, word_vectors=None, get_heads_next=False):
    """
    Eval all valid PP attachments from one (English) sentence

    max_distance is the maximum allowed distance of the candidate heads
    Current version only extracts candidate heads, prepositions, and first child of preposition
    """

    if not type(gold_sentence) == ConllSentence:
        print 'Warning: this method assumes sentence is ConllSentence, but it is instead:', str(type(gold_sentence))

    num_total = 0
    num_correct = 0
    for i in xrange(len(gold_sentence.poses)-1):
        if gold_sentence.is_prep(gold_sentence.poses[i]):
            children_indices = gold_sentence.get_children(i+1)
            for child_idx in children_indices:
                if gold_sentence.is_valid_attachment(i, child_idx-1, max_head_distance, max_child_distance):
                    gold_prep_parent = gold_sentence.parents[i]
                    pred_prep_parent = pred_sentence.parents[i]
                    heads = []
                    heads_next = []
                    prep = gold_sentence.tokens[i] if use_tokens else gold_sentence.lemmas[i]
                    child = gold_sentence.tokens[child_idx-1] if use_tokens else gold_sentence.lemmas[child_idx-1]
                    gold_head_label = -1
                    start = max(0, i - max_head_distance)
                    for j in xrange(start, i):  # find candidate heads
                        if gold_sentence.is_noun(gold_sentence.poses[j]) or gold_sentence.is_verb(gold_sentence.poses[j]):
                            if use_tokens:
                                heads.append(gold_sentence.tokens[j])
                            else:
                                heads.append(gold_sentence.lemmas[j])
                            if get_heads_next and j < i-1:
                                heads_next.append(gold_sentence.tokens[j+1])
                        if gold_prep_parent == j+1:  # if current candidate is the gold head
                            gold_head_label = len(heads)
                    if gold_head_label == -1:
                        sys.stderr.write('Error: could not find the gold head in sentence:\n' + str(gold_sentence))

                    if len(heads) > 1:  # don't take unambiguous cases
                        if get_heads_next:
                            attachment = ConllAttachment(heads, gold_head_label, prep, child, heads_next=heads_next)
                        else:
                            attachment = ConllAttachment(heads, gold_head_label, prep, child)
                        if word_vectors and attachment.has_word_vectors(word_vectors):
                            num_total += 1
                            if gold_prep_parent == pred_prep_parent:
                                num_correct += 1
    return num_total, num_correct


def get_pp_attachments_from_conll_sentence(sentence, max_head_distance, max_child_distance, use_tokens=True, word_vectors=None, get_heads_next=False, \
                                         use_heads_pos=False, use_heads_next_pos=False):
    """
    Get all valid PP attachments from one (English) sentence

    max_distance is the maximum allowed distance of the candidate heads
    Current version only extracts candidate heads, prepositions, and first child of preposition
    """

    if not type(sentence) == ConllSentence:
        print 'Warning: this method assumes sentence is ConllSentence, but it is instead:', str(type(sentence))

    attachments = []
    for i in xrange(len(sentence.poses)-1):
        if sentence.is_prep(sentence.poses[i]):
            children_indices = sentence.get_children(i+1)
            for child_idx in children_indices:
                if sentence.is_valid_attachment(i, child_idx-1, max_head_distance, max_child_distance):
                    gold_prep_parent = sentence.parents[i]
                    heads = []
                    heads_next = []
                    heads_pos = []
                    heads_next_pos = []
                    prep = sentence.tokens[i] if use_tokens else sentence.lemmas[i]
                    child = sentence.tokens[child_idx-1] if use_tokens else sentence.lemmas[child_idx-1]
                    gold_head_label = -1
                    start = max(0, i - max_head_distance)
                    for j in xrange(start, i):  # find candidate heads
                        if sentence.is_noun(sentence.poses[j]) or sentence.is_verb(sentence.poses[j]):
                            if use_tokens:
                                heads.append(sentence.tokens[j])
                            else:
                                heads.append(sentence.lemmas[j])
                            if use_heads_pos:
                                pos = sentence.poses[j]
                                pos_num = 1 if sentence.is_verb(pos) else -1  # verb=1, noun=-1
                                heads_pos.append(str(pos_num))
                            if get_heads_next and j < i-1:
                                heads_next.append(sentence.tokens[j+1])
                            if use_heads_next_pos and j < i-1:
                                heads_next_pos.append(sentence.poses[j+1])
                        if gold_prep_parent == j+1:  # if current candidate is the gold head
                            gold_head_label = len(heads)
                    if gold_head_label == -1:
                        sys.stderr.write('Error: could not find the gold head in sentence:\n' + str(sentence))

                    if len(heads) > 1:  # don't take unambiguous cases
                        if get_heads_next:
                            attachment = ConllAttachment(heads, gold_head_label, prep, child, heads_next=heads_next, heads_pos=heads_pos, heads_next_pos=heads_next_pos)
                        else:
                            attachment = ConllAttachment(heads, gold_head_label, prep, child, heads_pos=heads_pos, heads_next_pos=heads_next_pos)
                        if word_vectors and attachment.has_word_vectors(word_vectors):
                            attachments.append(attachment)
                        # attachments.append(attachment)
    return attachments


def extract_pp_attachments_from_conll_file(conll_filename, language, max_head_distance, max_child_distance, tokens=False, word_vectors=None, \
                                           get_heads_next=False, use_heads_pos=False, use_heads_next_pos=False):
    """
    Get all valid PP attachments from a file

    Return a list of attachments
    """

    print 'extracting attachments from file:', conll_filename
    sentences = read_conll_file(conll_filename, language)
    attachments = []
    for sentence in sentences:
        cur_attachments = get_pp_attachments_from_conll_sentence(sentence, max_head_distance, max_child_distance, tokens, word_vectors, \
                                                                 get_heads_next, use_heads_pos, use_heads_next_pos)
        attachments += cur_attachments
    return attachments


def eval_pp_attachments_from_conll_pred_file(conll_gold_filename, conll_pred_filename, language, max_head_distance, max_child_distance, \
                                             use_tokens=True, word_vectors=None, use_heads_next=False):

    print 'extracting attachments from file:', conll_gold_filename
    gold_sentences = read_conll_file(conll_gold_filename, language)
    pred_sentences = read_conll_file(conll_pred_filename, language)
    num_total = 0
    num_correct = 0
    if len(gold_sentences) != len(pred_sentences):
        print 'Error: len of gold and pred sentences not equal'
        return
    for i in xrange(len(gold_sentences)):
        cur_total, cur_correct = eval_pred_pp_attachments_from_conll_sentence(gold_sentences[i], pred_sentences[i], max_head_distance, \
                                                                              max_child_distance, use_tokens, word_vectors, use_heads_next)
        num_total += cur_total
        num_correct += cur_correct

    print 'total valid attachments:', num_total, 'correct:', num_correct, '(', "%.2f" %(100.0*num_correct/num_total), '%)'


def write_attachments(attachments, output_pref, get_heads_next=False):

    print 'writing attachments to files with prefix:', output_pref
    g_heads = open(output_pref + '.heads', 'w')
    g_nheads = open(output_pref + '.nheads', 'w')
    g_labels = open(output_pref + '.labels', 'w')
    g_pp_words = open(output_pref + '.ppwords', 'w')
    g_pp_parents = open(output_pref + '.ppparents', 'w')
    if get_heads_next:
        g_heads_next = open(output_pref + '.heads.next', 'w')

    for a in attachments:
        g_heads.write(' '.join(a.heads) + '\n')
        g_nheads.write(str(len(a.heads)) + '\n')
        g_labels.write(str(a.label) + '\n')
        g_pp_words.write(' '.join(a.pp_words) + '\n')
        g_pp_parents.write(' '.join([str(p) for p in a.pp_parents]) + '\n')
        if get_heads_next and a.heads_next:
            g_heads_next.write('\t'.join(a.heads_next) + '\n')

    g_heads.close()
    g_nheads.close()
    g_labels.close()
    g_pp_words.close()
    g_pp_parents.close()
    if get_heads_next:
        g_heads_next.close()


def write_wsj_attachments(attachments, output_pref, use_heads_next=False, use_heads_pos=False, use_heads_next_pos=False):

    write_conll_attachments(attachments, output_pref, use_heads_next, use_heads_pos, use_heads_next_pos)


def write_conll_attachments(attachments, output_pref, use_heads_next=False, use_heads_pos=False, use_heads_next_pos=False):

    print 'writing attachments to files with prefix:', output_pref
    g_heads = codecs.open(output_pref + '.heads.words', 'w', encoding='utf-8')
    g_nheads = open(output_pref + '.nheads', 'w')
    g_labels = open(output_pref + '.labels', 'w')
    g_preps = codecs.open(output_pref + '.preps.words', 'w', encoding='utf-8')
    g_children = codecs.open(output_pref + '.children.words', 'w', encoding='utf-8')
    if use_heads_next:
        g_heads_next = codecs.open(output_pref + '.heads.next.words', 'w', encoding='utf-8')
    if use_heads_pos:
        g_heads_pos = open(output_pref + '.heads.pos', 'w')
    if use_heads_next_pos:
        g_heads_next_pos = open(output_pref + '.heads.next.pos', 'w')

    for a in attachments:
        assert(isinstance(a, ConllAttachment))
        g_heads.write(' '.join(a.heads) + '\n')
        g_nheads.write(str(len(a.heads)) + '\n')
        g_labels.write(str(a.label) + '\n')
        g_preps.write(a.prep + '\n')
        g_children.write(a.child + '\n')
        if use_heads_next:
            g_heads_next.write('\t'.join(a.heads_next) + '\n')
        if use_heads_pos:
            g_heads_pos.write('\t'.join(a.heads_pos) + '\n')
        if use_heads_next_pos:
            g_heads_next_pos.write('\t'.join(a.heads_next_pos) + '\n')

    g_heads.close()
    g_nheads.close()
    g_labels.close()
    g_preps.close()
    g_children.close()
    if use_heads_next:
        g_heads_next.close()
    if use_heads_pos:
        g_heads_pos.close()
    if use_heads_next_pos:
        g_heads_next_pos.close()


def filter_attachments_by_max_children_num(attachments, max_child_count):

    print 'filtering attachments, max_child_count:', max_child_count
    filtered = []
    for a in attachments:
        child_count = np.bincount(np.bincount(a.pp_parents))
        if len(child_count) <= max_child_count+1:
            filtered.append(a)
    return filtered


def print_attachment_stats(attachments):

    print 'number of all attachments:', len(attachments)
    nheads = [len(a.heads) for a in attachments]
    print 'average # of heads:', np.mean(nheads), 'std:', np.std(nheads)
    nppwords = [len(a.pp_words) for a in attachments]
    vocab_heads = set()
    vocab_preps = set()
    vocab_children = set()
    vocab_child1 = set()
    vocab_all = set()
    for a in attachments:
        for h in a.heads:
            vocab_heads.add(h)
            vocab_all.add(h)
        vocab_preps.add(a.pp_words[0])
        vocab_all.add(a.pp_words[0])
        vocab_child1.add(a.pp_words[1])
        for w in a.pp_words[1:]:
            vocab_children.add(w)
            # vocab_all.add(w)
    print 'vocab sizes:'
    print 'heads:', len(vocab_heads), 'preps:', len(vocab_preps), 'children:', len(vocab_children), 'first child:', len(vocab_child1), 'all (head+prep+child1):', len(vocab_all)


def run_spmrl(spmrl_filename, output_pref, word_vectors_filename, max_distance, max_span, max_children=0, tokens=False, get_heads_next=False, only_child_grandchild=False):

    if only_child_grandchild and max_span > 0:
        sys.stderr.write('Error: cannot have positive max_span when only looking for child and grandchild (span ignored in this case)')
        return
    word_vectors = None
    # if len(word_vectors_filename):
    #     word_vectors = get_word_vectors(word_vectors_filename)
    attachments = extract_pp_attachments_from_file(spmrl_filename, max_distance, max_span, tokens, word_vectors, get_heads_next, only_child_grandchild)
    output_filename = output_pref
    if only_child_grandchild:
        output_filename += '.childgrandchild'
    else:
        if max_span > 0:
            output_filename += '.span' + str(max_span)
        if max_children > 0:  # if only attachments with no word having more than one child are wanted
            attachments = filter_attachments_by_max_children_num(attachments, max_children)
            output_filename += '.maxchildcount' + str(max_children)
    # write_attachments(attachments, output_filename, get_heads_next)
    print_attachment_stats(attachments)


def print_english_attachment_stats(attachments):

    print_conll_attachment_stats(attachments)


def print_conll_attachment_stats(attachments):

    print 'number of all attachments:', len(attachments)
    nheads = [len(a.heads) for a in attachments]
    print 'average # of heads:', np.mean(nheads), 'std:', np.std(nheads)
    vocab_heads = set()
    vocab_preps = set()
    vocab_child1 = set()
    vocab_all = set()
    for a in attachments:
        for h in a.heads:
            vocab_heads.add(h)
            vocab_all.add(h)
        vocab_preps.add(a.prep)
        vocab_all.add(a.prep)
        vocab_child1.add(a.child)
        vocab_all.add(a.child)
    print 'vocab sizes:'
    print 'heads:', len(vocab_heads), 'preps:', len(vocab_preps), 'first child:', len(vocab_child1), 'all (head+prep+child1):', len(vocab_all)
    print vocab_preps


def write_pp_predictions_to_wsj_file(attachments, wsj_dep_filename, pp_pred_filename, include_ind_filename):
    """
    Write predictions from PP model to conll-like file

    attachments - list of Attachment instances extracted from the conll_filename
    conll_filename - original conll_file, new file will be written to conll_filename.pred
    pp_pred_filename - file containing predictions from pp model, each line has number (1-indexed) of the predicted head (not word index; will be mapped to index)
    include_ind_filename - file containing indices of attachments included in pp model (1-indexing)
    """

    lines = open(wsj_dep_filename).readlines()
    preds = open(pp_pred_filename).readlines()
    preds = [int(pred.strip()) for pred in preds]
    orig_indices = open(include_ind_filename).readlines()
    orig_indices = [int(orig_index.strip()) for orig_index in orig_indices]
    # run some checks
    if len(attachments) < len(preds):
        sys.stderr.write('Error: number of extracted attachments cannot be smaller than predicted attachments from matlab code' + '\n')
        return
    if len(attachments) < max(orig_indices):
        sys.stderr.write('Error: maximum original index cannot be greater than number of extracted attachments' + '\n')
        return

    print 'writing pp predictions to new wsj file:', wsj_dep_filename + '.pred'

    # find predicted prep head indices
    map_prep_line_number_to_predicted_head_index = dict()
    for i in xrange(len(orig_indices)):
        orig_index = orig_indices[i]
        pred = preds[i]
        attachment = attachments[orig_index - 1]
        prep_line_number = attachment.sentence_start_line + attachment.orig_prep_id
        predicted_head_index = attachment.orig_heads_ids[pred-1] + 1
        map_prep_line_number_to_predicted_head_index[prep_line_number] = predicted_head_index
    # write new lines
    g = open(wsj_dep_filename + '.pred', 'w')
    for i in xrange(len(lines)):
        if i in map_prep_line_number_to_predicted_head_index:
            predicted_head_index = map_prep_line_number_to_predicted_head_index[i]
            new_line = lines[i].strip() + '\t' + str(predicted_head_index) + '\n'
        else:
            new_line = lines[i].strip() + '\t' + '_' + '\n'
        g.write(new_line)
    g.close()


def run_wsj(wsj_dep_filename, output_pref, word_vectors_filename, max_head_distance, max_child_distance, use_heads_next=False, \
            use_heads_pos=False, use_heads_next_pos=False, pp_pred_filename=None, include_ind_filename=None):

    word_vectors = None
    if len(word_vectors_filename):
        word_vectors = get_word_vectors(word_vectors_filename)
    attachments = extract_pp_attachments_from_wsj_dep_file(wsj_dep_filename, max_head_distance, max_child_distance, word_vectors, \
                                                           use_heads_next, use_heads_pos, use_heads_next_pos)
    write_wsj_attachments(attachments, output_pref, use_heads_next, use_heads_pos, use_heads_next_pos)
    print_english_attachment_stats(attachments)
    if pp_pred_filename and include_ind_filename:
        write_pp_predictions_to_wsj_file(attachments, wsj_dep_filename, pp_pred_filename, include_ind_filename)


def run_conll(conll_filename, language, output_pref, word_vectors_filename, max_head_distance, max_child_distance, use_tokens=True, use_heads_next=False, \
            use_heads_pos=False, use_heads_next_pos=False):
    word_vectors = None
    if len(word_vectors_filename):
        # if language == 'catalan':
        #     word_vectors = get_word_vectors_utf8(word_vectors_filename, 'latin-1')
        # else:
        word_vectors = get_word_vectors_utf8(word_vectors_filename)
        print len(word_vectors)
    attachments = extract_pp_attachments_from_conll_file(conll_filename, language, max_head_distance, max_child_distance, use_tokens, word_vectors, \
                                                           use_heads_next, use_heads_pos, use_heads_next_pos)
    write_conll_attachments(attachments, output_pref, use_heads_next, use_heads_pos, use_heads_next_pos)
    print_conll_attachment_stats(attachments)


SPMRL_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/spmrl/spmrl.train.lab.all'
OUTPUT_PREF = '/home/belinkov/Dropbox/school/pp/data/full-pps-matlab/arTenTen/spmrl.train.lab.all.pp'
WORD_VECTORS_FILE = '/mnt/scratch/belinkov/word2vec-arabic/dim25/vectors_arTenTen.concat.split.txt'
MAX_SPAN = 300
MAX_CHILDREN = 0
GET_HEADS_NEXT = False
ONLY_CHILD_GRANDCHILD = False
# run_spmrl(SPMRL_FILE, OUTPUT_PREF, WORD_VECTORS_FILE, 10, MAX_SPAN, MAX_CHILDREN, False, GET_HEADS_NEXT, ONLY_CHILD_GRANDCHILD)
WSJ_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/ptb/wsj.22.txt.dep'
OUTPUT_PREF = '/mnt/scratch/belinkov/pp/data/ptb/wsj.22.txt.dep.pp'
WORD_VECTORS_FILE = '/mnt/scratch/belinkov/word2vec-english/vectors_wiki/vectors_wiki_25.txt'
MAX_HEAD_DISTANCE = 10
MAX_CHILD_DISTANCE = 200
GET_HEADS_NEXT = False
GET_HEADS_POS = True
GET_HEADS_NEXT_POS = True
run_wsj(WSJ_FILE, OUTPUT_PREF, WORD_VECTORS_FILE, MAX_HEAD_DISTANCE, MAX_CHILD_DISTANCE, GET_HEADS_NEXT, GET_HEADS_POS, GET_HEADS_NEXT_POS)
# MATLAB_PRED_FILE = '/mnt/scratch/belinkov/pp/data/matlab/wsj.2-21.txt.dep.pp.out'
# MATLAB_INCLUDE_IND_FILE = '/mnt/scratch/belinkov/pp/data/matlab/wsj.2-21.txt.dep.pp.includeInd'
#run_wsj(WSJ_FILE, OUTPUT_PREF, WORD_VECTORS_FILE, MAX_HEAD_DISTANCE, MAX_CHILD_DISTANCE, GET_HEADS_NEXT, GET_HEADS_POS, GET_HEADS_NEXT_POS, \
#        MATLAB_PRED_FILE, MATLAB_INCLUDE_IND_FILE)


CONLL_FILE = '/mnt/scratch/belinkov/arabic-parser/data/conll/catalan/catalan07.test'
OUTPUT_PREF = '/mnt/scratch/belinkov/pp/data/catalan/catalan07.test.pp'
WORD_VECTORS_FILE = '/mnt/scratch/belinkov/word2vec-spanish/vectors_catalan_wiki_25.txt'
MAX_HEAD_DISTANCE = 10
MAX_CHILD_DISTANCE = 200
GET_HEADS_NEXT = False
GET_HEADS_POS = False
GET_HEADS_NEXT_POS = False
USE_TOKENS = False
# run_conll(CONLL_FILE, 'catalan', OUTPUT_PREF, WORD_VECTORS_FILE, MAX_HEAD_DISTANCE, MAX_CHILD_DISTANCE, USE_TOKENS)


SPMRL_CONLL_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/spmrl-conll/spmrl.train.lab.all.conll'
OUTPUT_PREF = '/mnt/scratch/belinkov/pp/data/spmrl-conll/spmrl.train.lab.all.conll.pp'
WORD_VECTORS_FILE = '/mnt/scratch/belinkov/word2vec-arabic/dim25/vectors_arTenTen.concat.split.txt'
MAX_HEAD_DISTANCE = 10
MAX_CHILD_DISTANCE = 200
GET_HEADS_NEXT = False
GET_HEADS_POS = True
GET_HEADS_NEXT_POS = True
USE_TOKENS = False
# run_conll(SPMRL_CONLL_FILE, 'arabic_spmrl', OUTPUT_PREF, WORD_VECTORS_FILE, MAX_HEAD_DISTANCE, MAX_CHILD_DISTANCE, \
#           USE_TOKENS, GET_HEADS_NEXT, GET_HEADS_POS, GET_HEADS_NEXT_POS)


### EVALS ###

# WORD_VECTORS_FILE = '/media/DATAPART1/backups/belinkov/word2vec-arabic/dim25/vectors_arTenTen.concat.split.txt'
# SPMRL_CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/rbg/spmrl.test.lab.1.conll'
# SPMRL_CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/rbg/spmrl.test.lab.1.conll.out'
# word_vectors = get_word_vectors(WORD_VECTORS_FILE)
# eval_pp_attachments_from_pred_file(SPMRL_CONLL_GOLD_FILE, SPMRL_CONLL_PRED_FILE, MAX_HEAD_DISTANCE, False, word_vectors, GET_HEADS_NEXT)
# SPMRL_CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/rbg/usepp/spmrl.test.lab'
# SPMRL_CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/rbg/usepp/spmrl.MST.usepp.out'
# eval_pp_attachments_from_pred_file(SPMRL_CONLL_GOLD_FILE, SPMRL_CONLL_PRED_FILE, MAX_HEAD_DISTANCE, False, word_vectors, GET_HEADS_NEXT)



#
# WORD_VECTORS_FILE = '/mnt/scratch/belinkov/word2vec-arabic/dim25/vectors_arTenTen.unlemmatized.split.unnormalized.unk.3.txt.rnn.utf8'
# word_vectors = get_word_vectors(WORD_VECTORS_FILE)
# # ATB_CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/rnn/2-Unvoc-Dev.utf8.txt.devtestswitch.merged.dep'
# # ATB_CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/rnn/2-Unvoc-Dev.utf8.txt.devtestswitch.out.merged.dep'
# ATB_CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/rnn/ATB-new/2-Unvoc-Dev.utf8.txt.merged.dep'
# ATB_CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/rnn/ATB-new/2-Unvoc-Dev.utf8.txt.out.merged.dep'
# eval_pp_attachments_from_stanford_atb_pred_file(ATB_CONLL_GOLD_FILE, ATB_CONLL_PRED_FILE, MAX_HEAD_DISTANCE, word_vectors, GET_HEADS_NEXT)
# # ATB_CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-first-stage-parser/ATB/2-Unvoc-Test.utf8.txt.top.S1.puncs.gold.dep'
# # ATB_CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-first-stage-parser/ATB/2-Unvoc-Test.utf8.txt.top.S1.puncs.out.cleaned.dep'
# # eval_pp_attachments_from_stanford_atb_pred_file(ATB_CONLL_GOLD_FILE, ATB_CONLL_PRED_FILE, MAX_HEAD_DISTANCE, word_vectors, GET_HEADS_NEXT)
# ATB_CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-first-stage-parser/ATB-new/2-Unvoc-Dev.utf8.txt.top.S1.puncs.gold.max70.dep'
# ATB_CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-first-stage-parser/ATB-new/2-Unvoc-Dev.utf8.txt.top.S1.puncs.out.cleaned.max70.dep'
# eval_pp_attachments_from_stanford_atb_pred_file(ATB_CONLL_GOLD_FILE, ATB_CONLL_PRED_FILE, MAX_HEAD_DISTANCE, word_vectors, GET_HEADS_NEXT)
# ATB_CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-rerank/ATB-new/2-Unvoc-Dev.utf8.txt.top.S1.puncs.gold.max70.dep'
# ATB_CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-rerank/ATB-new/2-Unvoc-Dev.utf8.txt.top.S1.puncs.out.max70.dep'
# eval_pp_attachments_from_stanford_atb_pred_file(ATB_CONLL_GOLD_FILE, ATB_CONLL_PRED_FILE, MAX_HEAD_DISTANCE, word_vectors, GET_HEADS_NEXT)
# ATB_CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-rs-arTenTen-ATB/2-Unvoc-Dev.utf8.txt.top.S1.puncs.gold.max70.dep'
# ATB_CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-rs-arTenTen-ATB/2-Unvoc-Dev.utf8.txt.top.S1.puncs.out.max70.dep'
# eval_pp_attachments_from_stanford_atb_pred_file(ATB_CONLL_GOLD_FILE, ATB_CONLL_PRED_FILE, MAX_HEAD_DISTANCE, word_vectors, GET_HEADS_NEXT)
# ATB_CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-rs-arTenTen-ATB3/2-Unvoc-Dev.utf8.txt.top.S1.puncs.gold.max70.dep'
# ATB_CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-rs-arTenTen-ATB3/2-Unvoc-Dev.utf8.txt.top.S1.puncs.out.max70.dep'
# eval_pp_attachments_from_stanford_atb_pred_file(ATB_CONLL_GOLD_FILE, ATB_CONLL_PRED_FILE, MAX_HEAD_DISTANCE, word_vectors, GET_HEADS_NEXT)



# WORD_VECTORS_FILE = '/media/DATAPART1/backups/belinkov/word2vec-english/vectors_wiki/vectors_wiki_25.txt'
# WSJ_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-rs/wsj.23.dep.txt'
# WSJ_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/charniak/charniak-rs-wikipedia-wsj3/wsj.23.txt.out.dep'
# word_vectors = get_word_vectors(WORD_VECTORS_FILE)
# eval_pp_attachments_from_wsj_pred_dep_file(WSJ_GOLD_FILE, WSJ_PRED_FILE, MAX_HEAD_DISTANCE, MAX_CHILD_DISTANCE, word_vectors, GET_HEADS_NEXT)
#
# WORD_VECTORS_FILE = '/mnt/scratch/belinkov/word2vec-spanish/vectors_catalan_wiki_25.txt'
# CONLL_GOLD_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/turbo/catalan07.test'
# CONLL_PRED_FILE = '/home/belinkov/Dropbox/school/arabic-parsing/data/turbo/catalan07.test.out'
# word_vectors = get_word_vectors_utf8(WORD_VECTORS_FILE)
# eval_pp_attachments_from_conll_pred_file(CONLL_GOLD_FILE, CONLL_PRED_FILE, 'catalan', MAX_HEAD_DISTANCE, MAX_CHILD_DISTANCE, USE_TOKENS, word_vectors, GET_HEADS_NEXT)
