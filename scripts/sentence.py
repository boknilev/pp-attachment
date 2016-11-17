# Sentence class

import sys

# for now, SPMRL tag set
POS_PREP = 'P'
POS_NOUN = 'N'
POS_VERB = 'V'
POS_PROPER_NOUN = 'PN'
POS_PUNC = 'PNX'
POS_CONJ = 'C'
POS_SEP_PRONOUN = 'PRO'
POS_SUF_PRONOUN = '_PRO'  # ends with this pattern (e.g. 3mp_PRO)
POS_ADJ = 'AJ'
MORPH_GENDER = 'G'
MORPH_NUMBER = 'N'
MORPH_PERSON = 'P'
ENGLISH_POS_TAGS = {'verb':['VB', 'VBD', 'VBN', 'VBP', 'VBZ'], 'noun':['NN', 'NNS'], 'prep':['IN', 'TO']}
ATB_POS_TAGS = ENGLISH_POS_TAGS  # for atb files preprocessed with stanford scripts (and converted to conll format with pennconverter)
CONLL_SPANISH_POS_TAGS = {'verb': ['va', 'vm', 'vs'], 'noun': ['nc'], 'prep': ['sp']}
CONLL_ARABIC_SPMRL_POS_TAGS = {'verb': ['V'], 'noun': ['N'], 'prep': ['P']}
CONLL_LANG_SPANISH = 'spanish'
CONLL_LANG_CATALAN = 'catalan'
CONLL_LANG_ARABIC_SPMRL = 'arabic_spmrl'  # for spmrl file converted to conll format (spmrl.*.conll)


class Sentence(object):
    """ Store a fully annotated sentence """

    tokens = []
    poses = []
    labels = [] # dependency labels, "---" means root
    parents = [] # indices of parents, starting with 1
                 # 0 means root
    morphs = []
    lemmas = []
    sentence_id = -1  # 0-based indexing

    def __init__(self, tokens, poses, labels, parents, morphs=[]):
        if len(tokens) != len(poses) or len(poses) != len(labels) or len(labels) != len(parents):
            print 'Error: bad arguments to Sentence.__init__'
            return
        self.tokens = tokens
        self.poses = poses
        self.labels = labels
        self.parents = parents
        self.morphs = morphs
        if self.morphs != []:
            # morph examples: L:|xar_1||C:g, L:mim~A_2
            #self.lemmas = self.get_lemmas(self.morphs)
            self.lemmas = [morph[2:] if morph.find('|') == -1 \
                            else morph[2:morph.find('|')] \
                                for morph in morphs]

    def __str__(self):
        res = '\t'.join(self.tokens) + '\n'
        res += '\t'.join(self.poses) + '\n'
        res += '\t'.join(self.labels) + '\n'
        res += '\t'.join([str(p) for p in self.parents]) + '\n'
        res += '\t'.join(self.morphs) + '\n'
        res += '\t'.join(self.lemmas) + '\n'
        return res

    def is_valid_attachment(self, prep_idx, max_distance):
        if abs(self.parents[prep_idx] - 1 - prep_idx) > max_distance:
            return False
        if self.parents[prep_idx] - 1 > prep_idx:  # skip left arcs
            return False
        parent_pos = self.poses[self.parents[prep_idx] - 1] if self.parents[prep_idx] > 0 else 'ROOT'
        if parent_pos != POS_VERB and parent_pos != POS_NOUN:  # only consider noun or verb parents
            return False
        if self.poses[prep_idx+1] != POS_NOUN:  # only consider prep-noun phrases (ignore e.g. prep-pronoun)
            return False
        return True

    def det_agree(self, idx1, idx2):
        if 'DT:t' in self.morphs[idx1] and 'DT:t' in self.morphs[idx2]:
            return True
        if 'DT:t' not in self.morphs[idx1] and 'DT:t' not in self.morphs[idx2]:
            return True
        return False

    def get_children(self, idx):
        """
        Get children for word in index idx

        idx starts with 1, zero meaning root
        """

        children = []  # will have indices with 1 indexing
        for i in xrange(len(self.parents)):
            if self.parents[i] == idx:
                children.append(i+1)
        return children

    def is_reachable(self, ancestor, node):
        """ Check if node is reachable from ancestor

        ancestor and node are 1-indexed
        """

        if ancestor == node:
            return True
        if node == 0:
            return False
        return self.is_reachable(ancestor, self.parents[node-1])

    def set_lemmas(self, lemmas):
        # use to set lemmas directly (not through initialization)
        self.lemmas = lemmas

    def set_start_line(self, start_line):
        self.start_line = start_line  # line where sentence starts (0-indexing)


class EnglishSentence(Sentence):

    # static class variables
    # ideally we'd do the same for the Sentence class (with SPMRL tags), but for now they are global variables
    POS_PREP = 'P'
    POS_NOUN = 'N'
    POS_VERB = 'V'

    #### methods for determining pos tags ####
    @staticmethod
    def is_prep(tag):
        for prep_tag in ENGLISH_POS_TAGS['prep']:
            if tag == prep_tag:
                return True
        return False

    @staticmethod
    def is_verb(tag):
        for verb_tag in ENGLISH_POS_TAGS['verb']:
            if tag == verb_tag:
                return True
        return False

    @staticmethod
    def is_noun(tag):
        for noun_tag in ENGLISH_POS_TAGS['noun']:
            if noun_tag == tag:
                return True
        return False

    def is_valid_attachment(self, prep_idx, child_idx, max_head_distance, max_child_distance):
        """
        here indices are 0-based
        """
        if self.parents[child_idx] - 1 != prep_idx:
            sys.stderr.write('Error: parent of child is not prep' + '\n')
            return False
        if abs(self.parents[prep_idx] - 1 - prep_idx) > max_head_distance:
            return False
        if abs(prep_idx - child_idx) > max_child_distance:
            return False
        if self.parents[prep_idx] - 1 > prep_idx:  # skip left arcs
            return False
        parent_pos = self.poses[self.parents[prep_idx] - 1] if self.parents[prep_idx] > 0 else 'ROOT'
        if not (self.is_verb(parent_pos) or self.is_noun(parent_pos)):  # only consider noun or verb parents
            return False
        if not self.is_noun(self.poses[child_idx]):  # only consider prep-noun phrases (ignore e.g. prep-pronoun)
            return False
        return True


class ATBSentence(Sentence):

    """
    Store an ATB sentence (preprocessed with stanford scripts and converted for conll format with pennconverter)
    """

    #### methods for determining pos tags ####
    @staticmethod
    def is_prep(tag):
        for prep_tag in ATB_POS_TAGS['prep']:
            if tag == prep_tag:
                return True
        return False

    @staticmethod
    def is_verb(tag):
        for verb_tag in ATB_POS_TAGS['verb']:
            if tag == verb_tag:
                return True
        return False

    @staticmethod
    def is_noun(tag):
        for noun_tag in ATB_POS_TAGS['noun']:
            if noun_tag == tag:
                return True
        return False

    def is_valid_attachment(self, prep_idx, max_distance):
        if abs(self.parents[prep_idx] - 1 - prep_idx) > max_distance:
            return False
        if self.parents[prep_idx] - 1 > prep_idx:  # skip left arcs
            return False
        parent_pos = self.poses[self.parents[prep_idx] - 1] if self.parents[prep_idx] > 0 else 'ROOT'
        if (not self.is_verb(parent_pos)) and (not self.is_noun(parent_pos)):  # only consider noun or verb parents
            return False
        if not self.is_noun(self.poses[prep_idx+1]):  # only consider prep-noun phrases (ignore e.g. prep-pronoun)
            return False
        return True

class ConllSentence(Sentence):
    """
    Store a fully annotated conll sentence
    """

    def __init__(self, tokens, poses, labels, parents, lemmas, language):
        if len(tokens) != len(poses) or len(poses) != len(labels) or len(labels) != len(parents):
            print 'Error: bad arguments to Sentence.__init__'
            return
        self.tokens = tokens
        self.poses = poses
        self.labels = labels
        self.parents = parents
        self.lemmas = lemmas
        self.language = language

    #### methods for determining pos tags ####
    @staticmethod
    def is_pos_lang(tag, pos_type, pos_map):
        for possible_tag in pos_map[pos_type]:
            if tag == possible_tag:
                return True
        return False

    def is_prep(self, tag):
        if self.language == CONLL_LANG_SPANISH or self.language == CONLL_LANG_CATALAN:
            return self.is_pos_lang(tag, 'prep', CONLL_SPANISH_POS_TAGS)
        elif self.language == CONLL_LANG_ARABIC_SPMRL:
            return self.is_pos_lang(tag, 'prep', CONLL_ARABIC_SPMRL_POS_TAGS)
        else:
            sys.stderr.write('Error: unsupported language ' + self.language + ' in ConllSentence')

    def is_verb(self, tag):
        if self.language == CONLL_LANG_SPANISH or self.language == CONLL_LANG_CATALAN:
            return self.is_pos_lang(tag, 'verb', CONLL_SPANISH_POS_TAGS)
        elif self.language == CONLL_LANG_ARABIC_SPMRL:
            return self.is_pos_lang(tag, 'verb', CONLL_ARABIC_SPMRL_POS_TAGS)
        else:
            sys.stderr.write('Error: unsupported language ' + self.language + ' in ConllSentence')

    def is_noun(self, tag):
        if self.language == CONLL_LANG_SPANISH or self.language == CONLL_LANG_CATALAN:
            return self.is_pos_lang(tag, 'noun', CONLL_SPANISH_POS_TAGS)
        elif self.language == CONLL_LANG_ARABIC_SPMRL:
            return self.is_pos_lang(tag, 'noun', CONLL_ARABIC_SPMRL_POS_TAGS)
        else:
            sys.stderr.write('Error: unsupported language ' + self.language + ' in ConllSentence')

    def is_valid_attachment(self, prep_idx, child_idx, max_head_distance, max_child_distance):
        if self.parents[child_idx] - 1 != prep_idx:
            sys.stderr.write('Error: parent of child is not prep' + '\n')
            return False
        if abs(self.parents[prep_idx] - 1 - prep_idx) > max_head_distance:
            return False
        if abs(prep_idx - child_idx) > max_child_distance:
            return False
        if self.parents[prep_idx] - 1 > prep_idx:  # skip left arcs
            return False
        parent_pos = self.poses[self.parents[prep_idx] - 1] if self.parents[prep_idx] > 0 else 'ROOT'
        if not (self.is_verb(parent_pos) or self.is_noun(parent_pos)):  # only consider noun or verb parents
            return False
        if not self.is_noun(self.poses[child_idx]):  # only consider prep-noun phrases (ignore e.g. prep-pronoun)
            return False
        return True
