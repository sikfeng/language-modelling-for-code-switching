import glob
import re
import jieba
from collections import defaultdict

jieba.enable_paddle()
jieba.initialize()

def convert_to_halfwidth(line):
    line = line.translate(str.maketrans("".join([chr(i) for i in range(ord('！'), ord('～') + 1)]), "".join([chr(i) for i in range(ord('!'), ord('~') + 1)]))).strip()
    line = line.translate(str.maketrans("《》、。“”", "(),.\"\"")).strip()
    return line

ch_radio_fnames = glob.glob("../datasets/man_ch_news_txt/ch_radio/cr??????", recursive=True)
p_daily_fnames = glob.glob("../datasets/man_ch_news_txt/p_daily/pd????", recursive=True)
xinhua_fnames = glob.glob("../datasets/man_ch_news_txt/xinhua/xh????_?", recursive=True)

with open("../extracted/raw_ch_radio_sents.txt", "w") as ch_radio_sents_f:
    for ch_radio_fname in ch_radio_fnames:
        with open(ch_radio_fname, encoding='gb18030') as ch_radio_f:
            while True:
                line = ch_radio_f.readline()
                if len(line) == 0:
                    break
                if line[:4] == "<HL>" and line[-5:] == "</HL>":
                    line = line[4:-5]
                    line = convert_to_halfwidth(line.strip())
                    line = " ".join(jieba.cut(line, use_paddle=True))
                    ch_radio_sents_f.write(line + '\n')
                    # tokens = jieba.cut(line[4:-5], use_paddle=True)
                    # for token in tokens:
                    #     w2i[token] += 1
                elif line.strip() == "<TEXT>":
                    sents = []
                    while True:
                        line = ch_radio_f.readline().strip()
                        if line.strip() == "</TEXT>":
                            break
                        if line != "<p>":
                            print("unexpected line that is not <p>")
                            print(line)
                            input("Press enter")
                        line = ch_radio_f.readline().strip()
                        for line_1 in re.split(r'[!?.]', line):
                            line_1 = convert_to_halfwidth(line_1.strip())
                            sents.append(line_1)

                    sents.pop()
                    for sent in sents:
                        sent = sent.translate(str.maketrans('', '', '#"\'%-`~()[]<>.,·¡™=§*?_/\\@&—;:'))
                        sent = " ".join(jieba.cut(sent, use_paddle=True))
                        if len(sent.strip()) == 0:
                            continue
                        ch_radio_sents_f.write(sent + '\n')
        print(f'Done with {ch_radio_fname}')

with open("../extracted/raw_p_daily_sents.txt", "w") as p_daily_sents_f:
    for p_daily_fname in p_daily_fnames:
        print(p_daily_fname)
        with open(p_daily_fname, encoding='gb18030') as p_daily_f:
            while True:
                line = p_daily_f.readline()
                if len(line) == 0:
                    break
                if line[:4] == "<HL>" and line[-5:] == "</HL>":
                    line = line[4:-5]
                elif line[0] == '<':
                    continue
                for line_1 in re.split(r'[.!?]', line):
                    line_1 = convert_to_halfwidth(line_1.strip())
                    line_1 = line_1.translate(str.maketrans('', '', '#"\'%-`~()[]<>.,·¡™=§*?_/\\@&—;:'))
                    if len(line_1.strip()) == 0:
                        continue
                    line_1 = " ".join(jieba.cut(line_1, use_paddle=True))
                    if len(line_1.strip()) == 0:
                        continue
                    p_daily_sents_f.write(line_1 + '\n')
        print(f'Done with {p_daily_fname}')

with open("../extracted/raw_xinhua_sents.txt", "w") as xinhua_sents_f:
    for xinhua_fname in xinhua_fnames:
        with open(xinhua_fname, encoding='gb18030') as xinhua_f:
            while True:
                line = xinhua_f.readline()
                if len(line) == 0:
                    break
                line = convert_to_halfwidth(line.strip())
                if line[:10] == "<headline>" and line[-11:] == "</headline>":
                    line = convert_to_halfwidth(line[10:-11].strip())
                elif line[:3] == "<s>" and line[-4:] == "</s>":
                    line = convert_to_halfwidth(line[3:-4].strip())
                else:
                    continue
                for line_1 in re.split(r'[.!?]', line):
                    line_1 = convert_to_halfwidth(line_1.strip())
                    if len(line_1.strip()) == 0:
                        continue
                    line_1 = line_1.translate(str.maketrans('', '', '#"\'%-`~()[]<>.,·¡™=§*?_/\\@&—;:'))
                    line_1 = " ".join(jieba.cut(line_1, use_paddle=True))
                    if line_1 == "完":
                        continue
                    if len(line_1.strip()) == 0:
                        continue
                    xinhua_sents_f.write(line_1 + '\n')
        print(f'Done with {xinhua_fname}')
