'''
Created on 7/18/18

@author: junkang

'''

import os,sys
import csv
import random
import numpy as np
import spacy
import pandas
import argparse
from progressbar import ProgressBar, FormatLabel, SimpleProgress, Percentage, Bar


INCLUDE_PUNCTUATION = True

class ROCStoriesEntity(object):

    MISSING_SENT_TAG = '<MISSING>'
    nlp = spacy.load('en_core_web_sm')

    def __init__(self, story, missing_idx):
        self.storyid = story['storyid']
        self.title = story['storytitle']
        self.sents = [story['sentence%d'%(i+1)] if i != missing_idx else self.MISSING_SENT_TAG for i in range(5)]
        self.missing_sent = story['sentence%d'%(missing_idx+1)]

        doc = self.nlp(self.missing_sent)

        self.missing_sent_tokens = [tok.text for tok in doc]
        self.missing_sent_tokens_pos = [tok.pos_ for tok in doc]

        self.missing_sent_tokens_np = [tok.text for tok in doc if not tok.is_punct]
        self.missing_sent_tokens_np_pos = [tok.pos_ for tok in doc if not tok.is_punct]

        missing_sent_tokens_idx = [(i, tok.text) for i, tok in enumerate(doc)]
        missing_sent_tokens_np_idx = [(i, tok.text) for i, tok in enumerate(doc) if not tok.is_punct]

        self.missing_sent_len = len(self.missing_sent_tokens)
        self.missing_sent_len_np = len(self.missing_sent_tokens_np)

        tokens = [(i, w, p) for (i, w), p in zip(missing_sent_tokens_idx, self.missing_sent_tokens_pos)]
        random.shuffle(tokens)

        tokens_no_punct = [(i, w, p) for (i, w), p in zip(missing_sent_tokens_np_idx, self.missing_sent_tokens_np_pos)]
        random.shuffle(tokens_no_punct)

        self.missing_sent_tokens_randomized = []
        self.missing_sent_tokens_randomized_pos = []
        self.missing_sent_tokens_randomized_idx = []
        for i, w, p in tokens:
            self.missing_sent_tokens_randomized.append(w)
            self.missing_sent_tokens_randomized_pos.append(p)
            self.missing_sent_tokens_randomized_idx.append(i)

        self.missing_sent_tokens_randomized_np = []
        self.missing_sent_tokens_randomized_np_pos = []
        self.missing_sent_tokens_randomized_np_idx = []
        for i, w, p in tokens_no_punct:
            self.missing_sent_tokens_randomized_np.append(w)
            self.missing_sent_tokens_randomized_np_pos.append(p)
            self.missing_sent_tokens_randomized_np_idx.append(i)

        self.accepted_words = []
        self.keywords = []

    def __str__(self):
        lines = []
        lines.append("title: %s"%self.title)

        for i, sent in enumerate(self.sents):
            lines.append("\tsent%d: %s"%(i+1, sent))

        lines.append("\tMissing sent: %s"%(self.missing_sent))
        lines.append("\tMissing sent's len: %d"%self.missing_sent_len)

        lines.append("\tAccepted Words: %s"%(self.accepted_words))
        lines.append("\tKeywords      : %s"%(self.keywords))

        # lines.append("\tMissing sent's tokens                : %s" % self.missing_sent_tokens)
        # lines.append("\tMissing sent's tokens(pos)           : %s" % self.missing_sent_tokens_pos)
        # lines.append("\tMissing sent's tokens(random)        : %s" % self.missing_sent_tokens_randomized)
        # lines.append("\tMissing sent's tokens(random, pos)   : %s" % self.missing_sent_tokens_randomized_pos)
        # lines.append("\tMissing sent's len_np: %d" % self.missing_sent_len_np)
        # lines.append("\tMissing sent's tokens_np             : %s" % self.missing_sent_tokens_np)
        # lines.append("\tMissing sent's tokens_np(pos)        : %s" % self.missing_sent_tokens_np_pos)
        # lines.append("\tMissing sent's tokens_np(random)     : %s" % self.missing_sent_tokens_randomized_np)
        # lines.append("\tMissing sent's tokens_np(random, pos): %s" % self.missing_sent_tokens_randomized_np_pos)

        return '\n'.join(lines)

    def row(self):
        """
        storyid, title, sent1, sent2, sent3, sent4, sent5, missing_sent_len, missing_sent, keywords, accepted_words
        :return:
        """
        row = [self.storyid, self.title]
        row.extend(self.sents)
        row.append(self.missing_sent_len)
        row.append(self.missing_sent)
        # row.append("||".join(self.missing_sent_tokens_randomized))
        # row.append("||".join(self.missing_sent_tokens_randomized_pos))
        # row.append("||".join(self.missing_sent_tokens_randomized_np))
        # row.append("||".join(self.missing_sent_tokens_randomized_np_pos))
        row.append("||".join(["%s:%d"%(tok,i) for i, tok in self.accepted_words]))
        row.append("||".join(self.keywords))
        return row

def load_rocstories(filepath):
    rocstories = []
    with open(filepath, 'r') as f:
        csvr = csv.DictReader(f)
        for story in csvr:
            rocstories.append(story)
    print("Loaded [%d] stories."%(len(rocstories)))
    return rocstories


def process_rocstories(stories):
    """
    - randomly select one missing sentence
    - randomly select accepted words: a list of (position, word)
    - randomly select keywords (excluding accepted words): a unordered list of words
    :param stories:
    :return:
    """
    story_size = len(stories)

    ''' Get missing sent indexes '''
    missing_sent_indexes = [random.randint(1,3) for _ in range(story_size)]
    hist, bins = np.histogram(missing_sent_indexes, bins=3)
    print("-"*80)
    print("Histogram of missing sent indexes")
    print(" ".join(["%5d" % b for b in bins[1:]]))
    print(" ".join(["%5d" % h for h in hist]))
    print("-" * 80)

    widgets = [FormatLabel('Processed: %(value)d stories (in: %(elapsed)s)'),Percentage(), " | ", SimpleProgress(), " | ", Bar()]
    pbar = ProgressBar(widgets=widgets)

    stories_processed = {l:[] for l in range(4, 14)}
    for i in pbar(range(len(stories))):
        story, missing_idx = stories[i], missing_sent_indexes[i]
        entity = ROCStoriesEntity(story, missing_idx)
        l = entity.missing_sent_len_np
        if l < 5:
            l = 4
        if l > 12:
            l = 13
        stories_processed[l].append(entity)

    a_lens = {}
    k_lens = {}
    for l, bucket in stories_processed.items():
        random.seed(l)

        a_lens[l] = []

        for entity in bucket:
            # select accepted words
            accepted_len = random.randint(0, l-KEYWORD_MIN-1)

            # # select keywords
            # keywords_len = l+1
            # while accepted_len + keywords_len > l:
            #     keywords_len = random.randint(0, l-1)

            # a_lens[l].append(accepted_len)
            # k_lens[l].append(keywords_len)

            # assert (accepted_len + keywords_len) <= l, "Missing sent ACPT/KEY selection: Something went wrong: [accepted:%d][keywords:%d][sent_len:%d]"%(accepted_len, keywords_len, l)

            entity.accepted_words = [(i, tok.lower()) for i, tok in zip(entity.missing_sent_tokens_randomized_np_idx, entity.missing_sent_tokens_randomized_np)][:accepted_len] if accepted_len > 0 else []
            # entity.keywords = [tok for tok in entity.missing_sent_tokens_randomized_np[accepted_len:keywords_len]] if keywords_len > 0 else []
            a_lens[l].append(len(entity.accepted_words))

    for l, bucket in stories_processed.items():
        random.seed(l)

        k_lens[l] = []
        entities_to_remove = []

        for entity in bucket:
            # select accepted words
            # accepted_len = random.randint(0, l-1)
            accepted_len = len(entity.accepted_words)

            # select keywords
            keywords_len = l+1
            while accepted_len + keywords_len > l:
                keywords_len = random.randint(KEYWORD_MIN, l-accepted_len)

            # A little nudge to push the distribution to the right
            if accepted_len + keywords_len < l:
                if random.random() > 0.5:
                    keywords_len += 1



            assert (accepted_len + keywords_len) <= l, "Missing sent ACPT/KEY selection: Something went wrong: [accepted:%d][keywords:%d][sent_len:%d]"%(accepted_len, keywords_len, l)

            entity.keywords = [tok for tok in entity.missing_sent_tokens_randomized_np[accepted_len:accepted_len+keywords_len]] if keywords_len > 0 else []
            if len(entity.keywords) == 0:
                print ("=> keywords_len: %d produced 0 keywords"%(keywords_len))
                print ("\t[awl: %d], src_token_len: %d (%s)"%(len(entity.accepted_words), len(entity.missing_sent_tokens_randomized_np), entity.title))
                entities_to_remove.append(entity)
                continue

            k_lens[l].append(len(entity.keywords))

        for e in entities_to_remove:
            stories_processed[l].remove(e)
            print("Removing: %s"%(e.title))

    for l in a_lens:
        a_bucket = a_lens[l]
        k_bucket = k_lens[l]
        # print_histogram(l, a_bucket, "Accepted words")
        # print_histogram(l, k_bucket, "Keywords")
        print_histogram2(l, a_bucket, k_bucket, "Accepted words",  "Keywords")

    return stories_processed



def export_html(filepath, rows, header, html_entity_limit):

    # def format_row(data, color="#ffa0a0"):
    #     attr = 'background-color: {}'.format(color)
    #     # print("- %s"%data[-1].split('%')[0])
    #     if float(data[-1].split('%')[0].split()[-1]) > MATCH_THRESHOLD:
    #         return [attr for _ in data]
    #     else:
    #         return ['' for _ in data]

    def format_pm_col(value):
        return '<div style="font-weight:bold;">{}</div>'.format(value)

    cutoff = html_entity_limit if html_entity_limit > 0 else len(rows)
    df = pandas.DataFrame().from_records(rows[:cutoff], columns=header)
    # html = df.style.apply(format_row, axis=1).render()
    # with open(filepath, 'w') as f:
    #     f.write(html)
    formatters = {'pm': format_pm_col}
    df.to_html(filepath, formatters=formatters, escape=False)

    print("Saved %d rows to [%s]"%(len(rows), filepath))

def print_histogram2(l, bucket1, bucket2, msg1, msg2):
    hist1, bins = np.histogram(bucket1, bins=[i for i in range(l + 1)])
    hist2, bins = np.histogram(bucket2, bins=[i for i in range(l + 1)])
    print("-" * 80)
    print("Histogram per sentences length (w/o punctuations) [%d]" % (l))
    print(" "*15          + " ".join(["%5d" % b for b in bins]))
    print("%15s" % (msg1) + " ".join(["%5d" % h for h in hist1]))
    print("%15s" % (msg2) + " ".join(["%5d" % h for h in hist2]))
    print("-" * 80)
    print("* Note that # of accepted words and keywords here are not grouped together. ex. An instance with 2 keywords could have any number of accepted_words.")
    print("")

def print_histogram(l, bucket, msg):
    hist, bins = np.histogram(bucket, bins=[i for i in range(l+1)])
    print("-"*80)
    print("Histogram for [%s:%s]"%(msg, l))
    print(" ".join(["%5d" % b for b in bins]))
    print(" ".join(["%5d" % h for h in hist]))
    print("-" * 80)
    print("")


def load_random_indexes(total_count, total_counts):
    filepath = "./rnd_%d_%d"%(total_count, len(total_counts))
    indexes = {}
    # for l in sorted(total_counts):
    #     print("%d: %d"%(l, total_counts[l]))

    # if os.path.exists(filepath) and False:
    #     with open(filepath, 'r') as f:
    #         for line in f:
    #             line = line.strip()
    #             toks = [int(i) for i in line.split(',')]
    #             indexes[toks[0]] = toks[1:]
    #     print("Loaded random indexes for %d entries (%d buckets) from [%s]"%(sum([len(v) for v in indexes.values()]),
    #                                                                          len(indexes), filepath))
    # else:
    for l, count in total_counts.items():
        indexes[l] = [i for i in range(count)]
        random.shuffle(indexes[l])
        # print("%d: min(%d), max(%d) < count[%d]"%(l, min(indexes[l]), max(indexes[l]), count))

    with open(filepath, 'w') as f:
        for l, index_list in sorted(indexes.items(), key=lambda items:items[0]):
            f.write(','.join([str(l)]+[str(i) for i in index_list])+'\n')
        print("Created random indexes for %d entries & saved to [%s]" % (total_count, filepath))

    return indexes


def save_dataset(filepath, dataset, header):
    with open(filepath, 'w') as f:
        csvw = csv.writer(f)
        csvw.writerow(header)
        for row in dataset:
            csvw.writerow(row.row())
    print("Saved %d rows to [%s]"%(len(dataset), filepath))



def export_datasets(stories, output_base_path, ratios):
    global DEBUG
    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)

    total_counts = {l:len(bucket) for l, bucket in stories.items()}
    total_count = sum(total_counts.values())
    rnd_indexes = load_random_indexes(total_count, total_counts)


    # print("")
    # for name, ratio in ratios:
    #     current_chunk_size = int(total_count * ratio)
    #
    #     indexes = [i for i in rnd_indexes[idx_offset:idx_offset + current_chunk_size]]
    #     hist, bins = np.histogram(indexes, bins=10)
    #     print("-"*80)
    #     print("Histogram of training dataset indexes:", name)
    #     print(" ".join(["%3d" % b for b in bins[1:]]))
    #     print(" ".join(["%3d" % h for h in hist]))
    #     print("-" * 80)
    #     idx_offset += current_chunk_size
    #
    # print("")

    count_table = [[0 for _ in stories] for _ in ratios]
    print("")
    print("Exporting Datasets for:", [n for n, _  in ratios])
    idx_offsets = {l:0 for l in stories} # per bucket idx offsets
    for ridx, (name, ratio) in enumerate(ratios):
        output_bucket = []
        for bidx, (l, bucket) in enumerate(sorted(stories.items(), key=lambda items:items[0])):
            idx_offset = idx_offsets[l]
            current_chunk_size = int(len(bucket) * ratio) if ridx < (len(ratios)-1) else len(bucket)-idx_offset

            # print("l:",l)
            # print("idx_offset:", idx_offset)
            # print("current_chunk_size:", current_chunk_size)
            # print("Bucket size : %s" % (len(bucket)))
            # print("rnd_indexes[l] : %s" % (len(rnd_indexes[l])))
            # sub_dataset = []
            # for i in rnd_indexes[l][idx_offset:idx_offset + current_chunk_size]:
            #     # print(i)
            #     sub_dataset.append(bucket[i])

            sub_dataset = [bucket[i] for i in rnd_indexes[l][idx_offset:idx_offset + current_chunk_size]]
            output_bucket.extend(sub_dataset)

            idx_offsets[l] += current_chunk_size
            # print("\t%5s[%d]: %6d/%6d(%.0f) => %d" % (name, l, current_chunk_size, len(bucket), ratio * 100, len(sub_dataset)))
            count_table[ridx][bidx] = current_chunk_size
        random.shuffle(output_bucket)
        output_path = os.path.join(output_base_path, name + '.csv')

        if DEBUG:
            output_path = output_path.replace(".csv", "_debug.csv")

        header = ['storyid', 'title', 'sent1', 'sent2', 'sent3', 'sent4', 'sent5', 'missing_sent_len', 'missing_sent',
                  'accepted_words', 'keywords']

        print("- %s: %d  => %s" % (name, len(output_bucket), output_path))
        save_dataset(output_path, output_bucket, header)
        print("")
        export_html(output_path.replace(".csv", ".html"), [r.row() for r in output_bucket], header, 200)





def createDataset():
    global DEBUG

    parser = argparse.ArgumentParser(description='Generate Collaborate Writing Dataset')

    parser.add_argument('-s', help="source (ROCStory) dataset file path", dest="src", default="/Users/junkang/Projects/ROCStories/ROCStories_winter2017 - ROCStories_winter2017.csv", type=str)
    parser.add_argument('-o', help="dataset output path", dest='output_path',
                        default='../dataset/', type=str)
    parser.add_argument('-debug', help="debug msg flag", dest="debug", action='store_true')

    args = parser.parse_args()

    src_filepath = args.src
    output_path = args.output_path
    DEBUG = args.debug

    pandas.set_option('display.max_colwidth', -1)

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'debug':
        DEBUG = True

    rocstories = load_rocstories(src_filepath)
    if DEBUG:
        rocstories = rocstories[:1000]
        print("Reduced story size down to [%d] for debugging"%(len(rocstories)))

    rocstories_processed = process_rocstories(rocstories)

    export_datasets(rocstories_processed, output_path, [('train', 0.50), ('valid', 0.1), ('test', 0.2), ('test_hold', 0.2)])

DEBUG = False
KEYWORD_MIN = 1

if __name__=='__main__':
    createDataset()