# This version extracts a more elaborate context for each PP attachment
import sys

from sentence import *
from utils import *


def get_pp_attachments_from_sentence(sentence, max_distance, ext=False, tokens=False):
    """
    Get all valid PP attachments from one sentence

    Each attachment is the parent index, followed by index-pos-morph triples of all words in a
    window of max_distance left to the prep, then the index-pos-morph triples of the prep its child

    If ext=True, include also nouns/adjectives following the prep's child, approximating a full PP
    If tokens=True, word forms will be extracted, otherwise morphs
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
                start = max(0, i - max_distance)
                context = [str(sentence.parents[i] - 1)]
                if tokens:
                    context += sentence.tokens[start:i+2]
                else:
                    context += [str(j) + '-' + sentence.poses[j] + '-' + sentence.morphs[j] + '-' + str(sentence.parents[j]-1) for j in range(start, i+2)] # \
                          #if sentence.poses[j] in [POS_NOUN, POS_VERB, POS_PREP]]
                if len(context) <= 4:  # ignore unambiguous attachments
                    continue

                if ext and i < len(sentence.poses) - 2:
                    if sentence.poses[i+2] == POS_ADJ:
                        # check agreement
                        if sentence.det_agree(i+1, i+2):
                            if tokens:
                                context.append(sentence.tokens[i+2])
                            else:
                                context.append(str(i+2) + '-' + sentence.poses[i+2] + '-' + sentence.morphs[i+2] + '-' + str(sentence.parents[i+2]-1))
                    elif sentence.poses[i+2] == POS_NOUN:
                        if not 'DT:t' in sentence.morphs[i+1]:
                            if tokens:
                                context.append(sentence.tokens[i+2])
                            else:
                                context.append(str(i+2) + '-' + sentence.poses[i+2] + '-' + sentence.morphs[i+2] + '-' + str(sentence.parents[i+2]-1))
                            if i < len(sentence.poses) - 3 and sentence.poses[i+3] == POS_ADJ:
                                if sentence.det_agree(i+2, i+3):
                                    if tokens:
                                        context.append(sentence.tokens[i+3])
                                    else:
                                        context.append(str(i+3) + '-' + sentence.poses[i+3] + '-' + sentence.morphs[i+3] + '-' + str(sentence.parents[i+3]-1))

                attachments.append(context)
    return attachments


def get_pred_pp_attachments_from_sentence(gold_sentence, pred_sentence, max_distance):
    """
    Get predictions for valid PP attachments from one sentence

    Determine valid attachments by gold sentence, extract decisions from pred sentence
    """

    attachments = []
    for i in xrange(len(gold_sentence.poses) - 1):
        if gold_sentence.poses[i] == POS_PREP:
            if gold_sentence.is_valid_attachment(i, max_distance):
                start = max(0, i - max_distance)
                context = [str(pred_sentence.parents[i] - 1)]
                context += [str(i) + '-' + gold_sentence.poses[i] + '-' + gold_sentence.morphs[i] + '-' + str(gold_sentence.parents[i]-1) for i in range(start, i+2)] # \
                          #if sentence.poses[i] in [POS_NOUN, POS_VERB, POS_PREP]]
                if len(context) <= 4:  # ignore unambiguous attachments
                    continue
                attachments.append(context)
    return attachments


def extract_pp_attachments_from_file(spmrl_filename, max_distance, ext=False, tokens=False):
    """
    Get all valid PP attachments from a file

    Return a list of lists of attachments
    """

    sentences = read_spmrl_file(spmrl_filename, True)
    attachments = []
    for sentence in sentences:
        cur_attachments = get_pp_attachments_from_sentence(sentence, max_distance, ext, tokens)
        attachments.append(cur_attachments)
    return attachments


def extract_pred_pp_attachments_from_file(gold_spmrl_filename, pred_spmrl_filename, max_distance):
    """
    Get all valid PP attachments with predicted decisions
    """

    gold_sentences = read_spmrl_file(gold_spmrl_filename, True)
    pred_sentences = read_spmrl_file(pred_spmrl_filename, False)
    if len(gold_sentences) != len(pred_sentences):
        sys.stderr.write('Error: incompatible gold and pred files')
        return
    attachments = []
    for i in xrange(len(gold_sentences)):
        cur_attachments = get_pred_pp_attachments_from_sentence(gold_sentences[i], pred_sentences[i], max_distance)
        attachments.append(cur_attachments)
    return attachments


def write_attachments(attachments, output_filename):
    """
    Write attachments to file
    """

    g = open(output_filename, 'w')
    for i in xrange(len(attachments)):
        g.write('# sentence ' + str(i) + '\n')
        for a in attachments[i]:
            g.write('\t'.join(a) + '\n')
    g.close()


def write_matlab_predictions(spmrl_filename, spmrl_pp_pred_filename):
    """
    Write matlab predictions to new spmrl-conll-like file

    spmrl_filename - spmrl.test.lab.1, spmrl.train.lab.all
    spmrl_pp_pred_filename - spmrl.test.lab.1.pp.pred, spmrl.train.lab.all.pp.pred
                             created by prepare_for_matlab.py
    """

    # attachments = extract_pp_attachments_from_file(spmrl_filename, 10, False, False)
    # attachment_prep_ids = []
    # for i in xrange(len(attachments)):
    #     cur_prep_ids = []
    #     for a in attachments[i]:
    #         prep_id = a[-1].split('_')[0]
    #         cur_prep_ids.append(prep_id)
    #     attachment_prep_ids.append(cur_prep_ids)

    # collect predictions in the same order as attachments are extracted
    pred_lines = open(spmrl_pp_pred_filename).readlines()
    preds = []
    cur_preds = []
    start = True
    for line in pred_lines:
        if start:
            start = False
            continue
        if line.startswith('# sentence'):
            preds.append(cur_preds)
            cur_preds = []
        else:
            cur_preds.append(line.strip())
    if cur_preds:
        preds.append(cur_preds)
        cur_preds = []


    # now extract attachments and write with predictions
    print 'writing matlab predicted attachments to conll-like file:', spmrl_filename + '.pred.conll'
    g = open(spmrl_filename + '.pred.conll', 'w')
    sentences = read_spmrl_file(spmrl_filename, True)
    for s in xrange(len(sentences)):
        sentence = sentences[s]
        attachments = get_pp_attachments_from_sentence(sentence, 10, False, False)
        cur_prep_ids = []
        for a in attachments:
            prep_id = int(a[-2].split('-')[0]) + 1  # correct index (was probably decreased during extraction)
            cur_prep_ids.append(prep_id)
        for i in xrange(len(sentence.tokens)):
            pred = '_'
            if i+1 in cur_prep_ids:
                loc = cur_prep_ids.index(i+1)
                predicted = preds[s][loc]
                if predicted != "#":
                    pred = str(int(predicted) + 1)  # again need to correct index
            # for now use pos for both cpos and pos
            g.write(str(i+1) + '\t' + sentence.tokens[i] + '\t' + sentence.lemmas[i] + '\t' + \
                    sentence.poses[i] + '\t' + sentence.poses[i] + '\t' + '_' + '\t' + str(sentence.parents[i]) + \
                    '\t' + sentence.labels[i] + '\t' + '_' + '\t' + '_' + '\t' + pred + '\n')
        g.write('\n')

    g.close()



GOLD_FILENAME = '/home/belinkov/Dropbox/school/arabic-parsing/data/spmrl/spmrl.train.lab.all'
OUTPUT_FILENAME = '/home/belinkov/Dropbox/school/pp/data/spmrl.train.lab.all.pp'
# attachments = extract_pp_attachments_from_file(GOLD_FILENAME, 10, False, False)
# write_attachments(attachments, OUTPUT_FILENAME)
GOLD_FILENAME = '/home/belinkov/Dropbox/school/arabic-parsing/data/spmrl/spmrl.train.lab.1'
PRED_FILENAME = '/home/belinkov/Dropbox/school/arabic-parsing/data/spmrl/crossval/spmrl.train.lab.1.crossval.out'
OUTPUT_FILENAME = '/home/belinkov/Dropbox/school/pp/data/spmrl.train.lab.1.crossval.out.pp'
# pred_attachments = extract_pred_pp_attachments_from_file(GOLD_FILENAME, PRED_FILENAME, 10)
# write_attachments(pred_attachments, OUTPUT_FILENAME)

GOLD_FILENAME = '/home/belinkov/Dropbox/school/arabic-parsing/data/spmrl/spmrl.train.lab.all'
PP_PRED_FILENAME = '/home/belinkov/Dropbox/school/pp/data/spmrl.train.lab.all.pp.pred'
write_matlab_predictions(GOLD_FILENAME, PP_PRED_FILENAME)