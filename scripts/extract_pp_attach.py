# extract cases of PP attachment from an MST file

import argparse

ATB_POS_TAGS = {'verb':['VB', 'VBD', 'VBN', 'VBP'], 'noun':['NN'], 'prep':['IN']}
SPMRL_POS_TAGS = {'verb':['V'], 'noun':['N', 'PN', 'AB'], 'prep':['P']}
ENGLISH_POS_TAGS = {'verb':['VB', 'VBD', 'VBN', 'VBP', 'VBZ'], 'noun':['NN'], 'prep':['IN', 'TO']}


class Sentence(object):
    """ Store a fully annotated sentence """

    tokens = []
    poss = []
    labels = [] # dependency labels, "---" means root
    parents = [] # indices of parents, starting with 1
                 # 0 means root
    morphs = []
    lemmas = []

    def __init__(self, lines, sentence_id, corpus_type):
        if len(lines) != 5:
            print 'Error: bad argument lines to Sentence.__init__'
            return
        for i in xrange(1,5):
            if len(lines[i].split('\t')) != len(lines[0].split('\t')):
                print 'Error: inconsistent line length in input lines ' + \
                       'for sentence', sentence_id, ':' \
                         + 'line', i, '!= line 0'
                print lines
                return
        self.sentence_id = sentence_id
        self.corpus_type = corpus_type
        for i in xrange(5):
            self.tokens = lines[0].strip().split('\t')
            self.poss = lines[1].strip().split('\t')
             # dependency labels, "---" means root
            self.labels = lines[2].strip().split('\t')
            # indices of parents, starting with 1
            # 0 means root
            
            parents_str = lines[3].strip().split('\t')
            self.parents = [int(parent) for parent in parents_str]
            self.morphs = lines[4].strip().split('\t')
            # morph examples: L:|xar_1||C:g, L:mim~A_2
            self.lemmas = self.get_lemmas(self.morphs)
            #self.lemmas = [morph[2:] if morph.find('|') == -1 \
            #                else morph[2:morph.find('|')] \
            #                    for morph in morphs]

        # count arcs direction {'left':count_left, 'right':count_right}
        count_left = 0
        count_right = 0
        for i in xrange(len(self.tokens)):
            if self.parents[i] == 0:
                # ignore roots
                continue
            # parents are 1-indexed
            if self.parents[i]-1 > i:
                count_left += 1
            elif self.parents[i]-1 < i:
                count_right += 1
            else:
                print 'Warning: parent = child'
        self.arcs_direction_count = {'left': count_left, 'right': count_right}

    def __str__(self):
        res = 'sentence id: ' + str(self.sentence_id) + ' corpus type: ' + self.corpus_type + '\n'
        res += '\t'.join(self.tokens) + '\n'
        res += '\t'.join(self.poss) + '\n'
        res += '\t'.join(self.labels) + '\n'
        res += '\t'.join([str(p) for p in self.parents]) + '\n'
        res += '\t'.join(self.morphs) + '\n'
        return res

    def get_pp_attachments(self):
        """ Find all instances of pp attachments """
 
        attachments = []
        indices = [] # indices of the words in each attachment,
                     # record verb, noun1, prep index 
                     # (noun2 always follows prep)
        for i in xrange(len(self.tokens)):
            verb = ''
            prep = ''
            noun = ''
            noun2 = ''  # child of prep
            noun2_ind = -1
            # search for a verb
            if self.is_verb(self.poss[i]):
                verb = self.lemmas[i]
                print 'found verb', verb, 'in position', i
                # search for a prep
                found_prep = False
                if i<len(self.tokens)-1 and self.is_verb(self.poss[i+1]):
                    continue # don't take first of two consecutive verbs
                j = i+2
                while j <= i+6 and j < len(self.tokens)-1:
                    if self.is_verb(self.poss[j]):  # don't allow another verb
                        break
                    if self.is_prep(self.poss[j]): 
                        if self.is_noun(self.poss[j+1]) and \
                          self.parents[j+1] == j+1:
                            # in Arabic, prep child follows prep
                            found_prep = True
                            prep = self.lemmas[j]
                            noun2 = self.lemmas[j+1]
                            noun2_ind = j+1
                            print 'found prep', prep, 'in position', j
                            break
                        elif self.corpus_type == 'english':
                            # in English, allow words between prep and child
                            found_noun_child = False
                            for l in xrange(len(self.parents)):
                                # find prep's child
                                if self.parents[l] == j+1 and \
                                   self.is_noun(self.poss[l]):
                                    found_prep = True
                                    prep = self.lemmas[j]
                                    noun2 = self.lemmas[l]
                                    noun2_ind = l
                                    print 'found prep', prep, 'in position', j
                                    print 'found noun2', noun2, 'in position', \
                                                                   noun2_ind
                                    found_noun_child = True
                                    break 
                            if found_noun_child:
                                break
                    j += 1    
                if found_prep:
                    noun = ''
                    found_noun = False
                    k = i+1
                    while k < j: 
                        # search for nouns
                        if self.is_noun(self.poss[k]):
                            found_noun = True
                            noun = self.lemmas[k]
                            print 'found noun', self.lemmas[k],\
                                  'in position', k
                            break
                        k += 1
                    if found_noun:
                        multiple_nouns = False
                        for t in xrange(k+1, j): 
                            # don't allow another noun between verb and prep
                            if self.is_noun(self.poss[t]):
                                print 'found another noun at:', t
                                print j
                                multiple_nouns = True
                        if not multiple_nouns:
                            print 'i', i, 'j', j, 'k', k
                            decision = ''
                            if self.parents[j] == k+1:
                                decision = '1' # noun parent
                            elif self.parents[j] == i+1:
                                decision = '0' # verb parent
                            if decision != '':
                                attachment = [decision, verb, noun, prep, noun2]
                                index = [decision, str(i), str(k), str(j), str(noun2_ind)]
                                # eliminate attachments with malformed words
                                is_legal_attachment = True
                                for w in attachment:
                                    if w.strip() == '':
                                        is_legal_attachment = False
                                if is_legal_attachment:
                                    attachments.append(attachment)
                                    indices.append(index)
            # end if
        # end for
        return attachments, indices

    def get_verb_prep_counts(self):
        """ Get counts for verb-prep dependencies """

        verb_prep_counts = dict()
        for i in xrange(len(self.tokens)):
            if self.is_verb(self.poss[i]):
                verb = self.lemmas[i]
                if verb.strip() == '': # empty verb
                    continue
                has_prep_child = False
                for j in xrange(len(self.tokens)):
                    if self.parents[j] == i+1 and \
                            self.is_prep(self.poss[j]):
                        has_prep_child = True
                        prep = self.lemmas[j]
                        if (verb, prep) in verb_prep_counts:
                            verb_prep_counts[(verb, prep)] += 1
                        else:
                            verb_prep_counts[(verb, prep)] = 1
                if not has_prep_child:
                    verb_prep_counts[(verb, 'na')] = 1
        return verb_prep_counts

    #### methods for determining pos tags for different data sets ####
    def is_prep(self, tag):
        if self.corpus_type == 'spmrl':
            if tag == SPMRL_POS_TAGS['prep'][0]:
                return True
        elif self.corpus_type == 'atb':
            if tag == ATB_POS_TAGS['prep'][0]:
                return True
        elif self.corpus_type == 'english':
            for prep_tag in ENGLISH_POS_TAGS['prep']:
                if tag == prep_tag:
                    return True
        return False

    def is_verb(self, tag):
        if self.corpus_type == 'spmrl':
            if tag == SPMRL_POS_TAGS['verb'][0]:
                return True
        elif self.corpus_type == 'atb':
            for verb_tag in ATB_POS_TAGS['verb']:
                if tag == verb_tag:
                    return True
        elif self.corpus_type == 'english':
            for verb_tag in ENGLISH_POS_TAGS['verb']:
                if tag == verb_tag:
                    return True
        return False

    def is_noun(self, tag):
        if self.corpus_type == 'spmrl':
            for noun_tag in SPMRL_POS_TAGS['noun']:
                if tag == noun_tag:
                    return True
        elif self.corpus_type == 'atb':
            for noun_tag in ATB_POS_TAGS['noun']:
                if noun_tag in tag:
                    return True
        elif self.corpus_type == 'english':
            for noun_tag in ENGLISH_POS_TAGS['noun']:
                if noun_tag in tag:
                    return True
        return False

    def is_pos(self, candidate_tag, pos_tag):
        if pos_tag == 'prep':
            return self.is_prep(candidate_tag)
        elif pos_tag == 'verb':
            return self.is_verb(candidate_tag)
        elif pos_tag == 'noun':
            return self.is_noun(candidate_tag)
        else:
            print 'Error: unknown pos tag in is_pos():', pos_tag
            return False
    #### end methods for pos tags ####

    def get_lemmas(self, morphs):
        if self.corpus_type == 'english':
            lemmas = morphs
        else:
            lemmas = [morph[2:] if morph.find('|') == -1 \
                             else morph[2:morph.find('|')] \
                                for morph in morphs]
        return lemmas

    def get_sub_sentence(self, sub_sentence_id, start, end):
        """ Make a sub-sentence """

        """
        start, end are 0-indexed, inclusive, here
        """

        new_lines = ['\t'.join(self.tokens[start:end]) + '\n']
        new_lines.append('\t'.join(self.poss[start:end]) + '\n')
        new_lines.append('\t'.join(self.labels[start:end]) + '\n')
        parents = []
        for p in self.parents[start:end]:
            # shift parent indices
            if p == 0:  # don't shift root
                parents.append(str(p))
            else:
                parents.append(str(p-start))
        new_lines.append('\t'.join(parents) + '\n')
        new_lines.append('\t'.join(self.morphs[start:end]) + '\n')
        #print new_lines
        sub_sentence = Sentence(new_lines, sub_sentence_id, self.corpus_type)
        return sub_sentence

    def write(self, out_file):
        out_file.write('\t'.join(self.tokens) + '\n')
        out_file.write('\t'.join(self.poss) + '\n')
        out_file.write('\t'.join(self.labels) + '\n')
        out_file.write('\t'.join([str(p) for p in self.parents]) + '\n')
        out_file.write('\t'.join(self.morphs) + '\n')
        out_file.write('\n')

    def is_projective(self):
        """ Check if the dependency tree is projective """

        for i in xrange(len(self.tokens)):
            child = i+1
            parent = self.parents[i]
            if parent == 0:
                # don't consider root nodes
                continue
            if parent < child:
                # right arc
                for j in xrange(parent+1, child):
                    if not self.is_reachable(parent, j):
                        return False
            elif parent > child:
                # left arc
                for j in xrange(child+1, parent):
                    if not self.is_reachable(parent, j):
                        return False
            else:  # parent = child
                print 'Error: parent = child'
        return True

    def is_reachable(self, ancestor, node):
        """ Check if node is reachable from ancestor """

        """
        ancestor and node are 1-indexed
        """

        if ancestor == node:
            return True
        if node == 0:
            return False
        return self.is_reachable(ancestor, self.parents[node-1])

#### end Sentence class ####


def process_mst_corpus(corpus_filename, output_filename, \
                                corpus_type):
    """ Extract pp attachments from a corpus in MST format """

    f = open(corpus_filename)
    lines = f.readlines()
    f.close()
    g = open(output_filename, 'w')
    g_ind = open(output_filename + '.ind', 'w')
    
    # sentence_block determines what mod to use for finding sentences
    sentence_block = 5 if corpus_type == 'english' else 6

    sentence_id = 0
    for i in xrange(len(lines)-sentence_block):
        if i % sentence_block == 0:
            sentence_id += 1
            if corpus_type == 'english':
                # lemmas are the tokens
                sentence_lines = lines[i:i+sentence_block-1]
                sentence_lines.append(lines[i])
            else:
                sentence_lines = lines[i:i+sentence_block-1]
            sentence = Sentence(sentence_lines, sentence_id, corpus_type)
            attachments, indices = sentence.get_pp_attachments()
            if len(attachments) != len(indices):
                print 'Warning: incompatible number of attachments and indices'
                continue
            for t in xrange(len(attachments)):
                g.write('\t'.join(attachments[t]) + '\n')
                g_ind.write(str(sentence_id) + '\t' + '\t'.join(indices[t]) + '\n')
    g.close()
    g_ind.close()

"""
tokens_line = 'w	qtl	qA}d	AlTA}rp	w	msAEd	h	ElY	Alfwr	ADAfp	AlY	vlAvp	mdnyyn	ElY	AlArD	w	jrH	Edd	|xr	.'
poss_line = 'CC	VBN	NN	DT+NN	CC	NN	PRP$	IN	DT+NN	NN	IN	NN	NNS	IN	DT+NN	CC	VBN	NN	JJ	PUNC'
labels_line = 'MOD	---	SBJ	IDF	MOD	OBJ	IDF	MOD	OBJ	MOD	MOD	OBJ	IDF	MOD	OBJ	MOD	OBJ	SBJ	MOD	MOD'
parents_line = '2	0	2	3	3	5	6	2	8	2	10	11	12	12	14	2	16	17	18	2'
morphs_line = 'L:wa	L:qatal-u_1||G:m||P:3||N:s	L:qA}id_1||G:m||N:s||C:n||S:c	L:TA}irap_1||G:f||N:s||C:g||S:d	L:wa	L:musAEid_1||G:m||N:s||C:n||S:c	L:hu||G:m||P:3||N:s	L:EalaY_1	L:fawor_1||G:m||N:s||C:g||S:d	L:<iDAfap_1||G:f||N:s||C:a||S:i	L:<ilaY_1	L:valAv_1||G:f||N:s||C:g||S:c	L:madaniy~_1||G:m||N:p||C:g||S:i	L:EalaY_1	L:>aroD_1||G:m||N:s||C:g||S:d	L:wa	L:jaraH-a_1||G:m||P:3||N:s	L:Eadad_1||G:m||N:s||C:n||S:i	L:|xar_1||G:m||N:s||C:n||S:c	L:.'
"""
"""
tokens_line = 'w	wqEtA	EddA	mn	AlAtfAqAt	fy	mjAl	AltEAwn	AlsyAHy	.'
poss_line = 'CC	VBD	NN	IN	DT+NNS	IN	NN	DT+NN	DT+JJ	PUNC'
labels_line = 'MOD	---	OBJ	MOD	OBJ	MOD	OBJ	IDF	MOD	MOD'
parents_line = '2	0	2	3	4	5	6	7	8	2'
morphs_line = 'L:wa	L:waq~aE_1||G:f||P:3||N:d	L:Eadad_1||G:m||N:s||C:a||S:i	L:min_1	L:{it~ifAq_1||G:f||N:p||C:g||S:d	L:fiy_1	L:majAl_1||G:m||N:s||C:g||S:c	L:taEAwun_1||G:m||N:s||C:g||S:d	L:siyAHiy~_1||G:m||N:s||C:g||S:d	L:.'
"""
"""
# spmrl example
tokens_line = '<num>	-	bljykA	:	Almlk	lywbwld	AlvAny	ytnAzl	En	AlEr$	l	Abn	h	bwdwAn	bEd	Hrkp	$Ebyp	wAsEp	TAlbt	b	Ezl	h	.'
poss_line = 'N_NUM	PNX	PN	PNX	N	PN	PN	V	P	N	P	N	3ms_PRO	PN	N	N	AJ	AJ	V	P	N	3ms_PRO	PNX'
labels_line = '---	MOD	MOD	MOD	SBJ	MOD	---	---	MOD	OBJ	MOD	OBJ	IDF	MOD	MOD	IDF	MOD	MOD	MOD	MOD	OBJ	IDF	MOD'
parents_line = '0	1	1	1	8	5	6	0	8	9	8	11	12	12	8	15	16	16	16	19	20	21	8'
morphs_line = 'L:<num>||G:m||N:s||FG:B||FN:P||R:N	L:-	L:bljykA||G:m||N:s||FG:F||FN:S||R:I	L:=	L:mlk||G:m||N:s||FG:M||FN:S||R:R||DT:t	L:lywbwld||G:m||N:s||FG:M||FN:S||R:R	L:vAny||G:m||N:s||FG:M||FN:S||R:R||DT:t	L:tnAzl||G:m||N:s||FG:M||FN:S||P:3||R:N	L:En	L:Er$||G:m||N:s||FG:M||FN:S||R:I||DT:t	L:l+	L:Abn||G:m||N:s||FG:M||FN:S||R:R	L:h||FG:F||FN:S||R:N	L:bwdwAn||G:m||N:s||FG:M||FN:S||R:R	L:bEd||G:m||N:s	L:Hrkp||G:f||N:s||FG:F||FN:S||R:I	L:$Eby||G:f||N:s||FG:F||FN:S||R:I	L:wAsE||G:f||N:s||FG:F||FN:S||R:N	L:TAlb||G:f||N:s||FG:F||FN:S||P:3||R:N	L:b+	L:Ezl||G:m||N:s||FG:M||FN:S||R:I	L:h||FG:F||FN:S||R:N	L:.'
"""
"""
# english example
tokens_line = 'Apogee	said	a	subcontractor	had	severe	cost	overruns	and	was	unable	to	fulfill	the	contract	terms	on	its	own	,	making	it	necessary	for	Apogee	to	advance	cash	to	ensure	completion	of	the	project	.'
poss_line = 'NNP	VBD	DT	NN	VBD	JJ	NN	NNS	CC	VBD	JJ	TO	VB	DT	NN	NNS	IN	PRP$	JJ	,	VBG	PRP	JJ	IN	NNP	TO	VB	NN	TO	VB	NN	IN	DT	NN	.'
labels_line = 'SBJ	ROOT	NMOD	SBJ	COORD	NMOD	NMOD	OBJ	OBJ	COORD	VMOD	VMOD	AMOD	NMOD	NMOD	OBJ	ADV	NMOD	PMOD	P	ADV	SBJ	VMOD	VMOD	SBJ	VMOD	OBJ	IOBJ	VMOD	OBJ	OBJ	NMOD	NMOD	PMOD	P'
parents_line = '2	0	4	9	9	8	8	5	2	9	10	13	11	16	16	13	13	19	17	10	10	27	27	27	27	27	21	27	30	27	30	31	34	32	2'
morphs_line = 'Apogee	said	a	subcontractor	had	severe	cost	overruns	and	was	unable	to	fulfill	the	contract	terms	on	its	own	,	making	it	necessary	for	Apogee	to	advance	cash	to	ensure	completion	of	the	project	.'
"""
"""
lines = [tokens_line, poss_line, labels_line, parents_line, morphs_line]
                        
sentence = Sentence(lines, 1, 'english')
print sentence.tokens
print sentence.poss
print sentence.labels
print sentence.parents
print sentence.lemmas
                   
attachments = sentence.get_pp_attachments()                
print attachments
counts = sentence.get_verb_prep_counts()
print counts
print sentence.arcs_direction_count
"""

#"""
def main():
    parser = argparse.ArgumentParser(description='Extract PP attachments')
    parser.add_argument('corpus_file', help='Dependecy sentences in MST format')
    parser.add_argument('output_file', help='File to write extracted attachments')
    parser.add_argument('-t', '--corpus_type', choices=['atb', 'spmrl', 'english'],\
                        default='atb', help='Specify type of corpus')
    args = parser.parse_args()
    process_mst_corpus(args.corpus_file, args.output_file, args.corpus_type)


if __name__=='__main__':
    main()         
#"""
            
