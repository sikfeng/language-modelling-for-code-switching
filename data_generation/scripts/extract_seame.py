import glob
import re
import jieba
from collections import defaultdict

jieba.enable_paddle()
jieba.initialize()

zh_regex = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)')
en_regex = re.compile(r'([a-zA-Z])+')

num_cs_sents = 0
num_zh_sents = 0
num_en_sents = 0

zh_words = defaultdict(int)
en_words = defaultdict(int)


def clean_line(line):
    line = " ".join(line.split()[3:]).strip() # remove audio file info (since we dont use it)
    line = re.sub(r"[\(\[<{（【［].*?[\)\]>}）】］]", "", line).strip() # remove all the hesitation sounds, non verbal sounds
    #line = re.sub(r'\(ppl\)|\(ppb\)|\(ppo\)|<unk>|[hmm]|(em)', '', line).strip()
    line = re.sub(r"[\u0080-\u00ff]", "", line).strip() # remove some invalid characters in the transcriptions
    line = line.translate(str.maketrans("".join([chr(i) for i in range(ord('！'), ord('～') + 1)]), "".join([chr(i) for i in range(ord('!'), ord('~') + 1)]))).strip() # convert to half width
    line = re.sub(r'([a-zA-Z]\.[\s]+)+', lambda sub: "".join(sub.group().split()), line).strip() # put abbreviations as one token together
    line = line.translate(str.maketrans('', '', '#"\'%-`~()[]<>.,·¡™=§*?_/\\@&—'))
    return line


def tokenize_sent(sent):
    sent = " ".join(" ".join(jieba.cut(sent, use_paddle=True)).split())
    tokens = sent.split()
    lang = tokens[0]
    seperated_tokens = []
    for token in tokens[1:]:
        seperated_tokens += re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[0-9]+|[^0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\s]+", token, re.UNICODE)
    return lang, seperated_tokens


with open('../extracted/seame_sents.txt', 'w') as sents_f:
    for transcript_fname in glob.glob("../datasets/seame/**/phaseII/*.txt", recursive=True):
        print(transcript_fname)
        with open(transcript_fname, 'r') as transcript_f:
            for line in transcript_f:
                zh = False
                en = False

                line = clean_line(line)
                lang, tokens = tokenize_sent(line)

                for idx, token in enumerate(tokens):
                    token = token.strip()
                    if zh_regex.match(token):
                        zh = True
                        token += "__zh"
                    elif en_regex.match(token):
                        en = True
                        token += "__en"
                    elif token.isdigit():
                        continue
                    else:
                        print(f'broken token: {token}')
                        continue

                    tokens[idx] = token

                if len(tokens) == 0:
                    continue
                        
                if zh and en:
                    if lang != "CS":
                        print(f'{tokens} is CS but is labeled {lang}')
                    num_cs_sents += 1
                elif zh:
                    if lang != "ZH":
                        print(f'{tokens} is ZH but is labeled {lang}')
                    num_zh_sents += 1
                elif en:
                    if lang != "EN":
                        print(f'{tokens} is EN but is labeled {lang}')
                    num_en_sents += 1
                else:
                    print(f'broken sentence: {tokens}')
                    continue

                sents_f.write(" ".join(tokens) + '\n')

                for token in tokens:
                    token = token.strip()
                    if token[-4:] == "__zh":
                        zh_words[token[:-4]] += 1
                    elif token[-4:] == "__en":
                        en_words[token[:-4]] += 1
                    elif token.isdigit():
                        continue
                    else:
                        print(f'broken token: {token}')
                        # ignore 
                        continue

print("Done extracting sentences")
print(f'{num_cs_sents} CS sentences, {num_zh_sents} Chinese sentences, {num_en_sents} English sentences.')
