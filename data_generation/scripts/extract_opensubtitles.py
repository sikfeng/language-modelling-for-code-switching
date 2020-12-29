import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm


def is_chinese(token):
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    for cp in token:
        if cp in "^&*{}[]\|<>/":
            return False
        if cp.isdigit() or cp in ",. ":
            continue
        else:
            cp = ord(cp)
            if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
            ):  #
                continue
            else:
                return False
    return True


def is_english(token):
    if all(c.isdigit() or c in "-,. " or (ord(c) <= ord('z') and ord(c) >= ord('a')) or (ord(c) <= ord('Z') and ord(c) >= ord('A')) for c in token):
        return True
    return False


def check_sent(sent):
    # this is so overcomplicated, i should probably simplify it
    filter_chars = "^&*{}[]\|<>/"
    if len(sent) < 4:
        return False
    zh_words = 0
    en_words = 0
    for word in sent:
        for filter_char in filter_chars:
            if filter_char in word:
                return False
        if is_chinese(word):
            zh_words += 1
        elif is_english(word):
            en_words += 1
        if en_words > 0 and zh_words > 0:
            return False
    return True


def clean_sent(sent):
    for idx, word in enumerate(sent):
        if is_chinese(word):
            sent[idx] += "__zh"
        elif is_english(word):
            sent[idx] += "__en"
    sent = " ".join(sent)
    if sent[:2] == "- ":
        sent = sent[2:]
    if sent[0] == "-":
        sent = sent[1:]
    while sent[:4] == "__zh" or sent[:4] == "__en":
        sent = sent[5:]
    return sent.lower()


def lang(sent):
    en = "__en" in sent
    zh = "__zh" in sent
    if en and zh:
        return "cs"
    elif en:
        return "en"
    elif zh:
        return "zh"
    else:
        print("ERROR!")


def main():
    # set a limit on how many files to extract since we can't even generate that many sentences in a reasonable time
    subtitle_dirs = glob.glob("../datasets/OpenSubtitles/xml/en/*/*")[:10000] + glob.glob("../datasets/OpenSubtitles/xml/zh_cn/*/*")[:10000]

    to_process = []

    for subtitle_dir in tqdm(subtitle_dirs):
        subtitle_fs = glob.glob(subtitle_dir + "/*")
        subtitle_f = sorted(subtitle_fs)[0]
        to_process.append(subtitle_f)

    print(f"{len(to_process)} files to process")

    sents = []
    for subtitle_f in tqdm(to_process):
        try:
            tree = ET.parse(subtitle_f)
            root = tree.getroot()
            for child in root:
                if child.tag == 's':
                    sent = [word.text for word in child if word.tag == 'w']
                    if check_sent(sent):
                        sent = clean_sent(sent)
                        if len(sent) > 0: 
                            sents.append(sent)
        except:
            continue

    print(f"{len(sents)} extracted")

    en_num = 0
    zh_num = 0
    with open('../extracted/opensubtitles_sents.txt', 'w') as sents_f:
        for sent in sents:
            sents_f.write(sent + '\n')
            if lang(sent) == "en":
                en_num += 1
            elif lang(sent) == "zh":
                zh_num += 1
            else:
                print("ERROR")
    print(en_num, zh_num)
    return


main()

