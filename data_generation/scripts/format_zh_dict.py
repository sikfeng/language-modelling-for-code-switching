import pinyin
import re

# code here largely adapted from 
# https://github.com/kaldi-asr/kaldi/blob/master/egs/hkust/s5/conf/pinyin2cmu
# https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/pinyin_map.pl


pinyin_map = {"A": "AA",
              "AI": "AY",
              "AN": "AE N",
              "ANG": "AE NG",
              "AO": "AW",
              "B": "B",
              "CH": "CH",
              "C": "T S",
              "D": "D",
              "E": "ER",
              "EI": "EY",
              "EN": "AH N",
              "ENG": "AH NG",
              "ER": "AA R",
              "F": "F",
              "G": "G",
              "H": "HH",
              "IA": "IY AA",
              "IANG": "IY AE NG",
              "IAN": "IY AE N",
              "IAO": "IY AW",
              "IE": "IY EH",
              "I": "IY",
              "ING": "IY NG",
              "IN": "IY N",
              "IONG": "IY UH NG",
              "IU": "IY UH",
              "J": "J",
              "K": "K",
              "L": "L",
              "M": "M",
              "N": "N",
              "O": "AO",
              "ONG": "UH NG",
              "OU": "OW",
              "P": "P",
              "Q": "Q",
              "R": "R",
              "SH": "SH",
              "S": "S",
              "T": "T",
              "UAI": "UW AY",
              "UANG": "UW AE NG",
              "UAN": "UW AE N",
              "UA": "UW AA",
              "UI": "UW IY",
              "UN": "UW AH N",
              "UO": "UW AO",
              "U": "UW",
              "UE": "IY EH",
              "VE": "IY EH",
              "V": "IY UW",
              "VN": "IY N",
              "W": "W",
              "X": "X",
              "Y": "Y",
              "ZH": "JH",
              "Z": "Z"}


def pinyin_to_phoneme(pinyin_tokens):
    pinyin_tokens = pinyin_tokens.replace("U:", "V") # make easier to format
    pinyin_tokens = pinyin_tokens.translate(str.maketrans("", "", ",·")) # remove punctuations
    pinyins = pinyin_tokens.split()
    converted = []
    for pinyin in pinyins:
        if re.match("R[0-9]", pinyin):
            pinyin = "ER1"
        if re.match("XX[0-9]?", pinyin):
            # some word with unknown pronounciation in cedict, so ignore
            return "failed"
        suffix = ""
        for prefix in ["CH", "SH", "ZH", "B", "C", "D", "F", "G", "H", "J", "K", 
                       "L", "M", "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"]:
            match = re.match(prefix + "([A-Z]+)([0-9])", pinyin)
            if match:
                suffix = match.group(1)
                tone = match.group(2)
                tone = "" 
                converted.append(pinyin_map[prefix])
                converted += [s + tone for s in pinyin_map[suffix].split()]
                break
        if suffix:
            continue
        # if not mached to a prefix
        match = re.match("([A-Z]+)([0-9])?", pinyin)
        suffix = match.group(1)
        tone = match.group(2)
        if tone is None:
            tone = ""
        tone = "" ####
        converted += [s + tone for s in pinyin_map[suffix].split()]
    return converted

zh_words = dict()

with open("../extracted/cedict_1_0_ts_utf-8_mdbg.txt", 'r') as dict_zh_f:
    for line in dict_zh_f:
        if line[0] == '#':
            continue
        line = line.split('/', 1)[0].strip() # ignore the definitions
        _, zh_word, word_pinyin = line.split(" ", 2) # use the simplified words, ignore traditional
        zh_words[zh_word] = word_pinyin.strip().upper()[1:-1] # convert pinyin to uppercase, remove bracket

print("Done extracting words from cedict")

with open("../extracted/zh_dict.txt", 'w') as zh_dict_f:
    for zh_word, pinyin in zh_words.items():
        phonemes = pinyin_to_phoneme(pinyin)
        if phonemes == "failed":
            continue
        zh_dict_f.write(f'{zh_word}\t')
        zh_dict_f.write(" ".join(phonemes))
        zh_dict_f.write("\n")

print("Done writing dictionary")
