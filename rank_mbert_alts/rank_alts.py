import json

def lang(s):
    en = "__en" in s
    zh = "__zh" in s
    if en and zh:
        return "cs"
    if en:
        return "en"
    if zh:
        return "zh"
    return "error!"

dev_set = json.load(open("alternate_sents_seame_val_filtered.json"))
dev_scores = json.load(open("dev_scores.json"))

test_set = json.load(open("alternate_sents_seame_test_filtered.json"))
test_scores = json.load(open("test_scores.json"))

def get_ranks(eval_set, eval_scores):
    eval_stats = {'cs': [0]*6, 'en': [0]*6, 'zh': [0]*6}
    for alt_set, scores in zip(eval_set, eval_scores):
        lang_scores = {'cs': [], 'en': [], 'zh': []}
        for sent, score in zip(alt_set["cs_alternatives"] + alt_set["en_alternatives"] + alt_set["zh_alternatives"], scores[1:]):
            lang_scores[lang(sent)].append(score[0])
        avg_cs = sum(lang_scores['cs'])/len(lang_scores['cs'])
        avg_en = sum(lang_scores['en'])/len(lang_scores['en'])
        avg_zh = sum(lang_scores['zh'])/len(lang_scores['zh'])
    
        if avg_cs > avg_en > avg_zh:
            eval_stats[lang(alt_set["orig"])][0] += 1
        if avg_cs > avg_zh > avg_en:
            eval_stats[lang(alt_set["orig"])][1] += 1
        if avg_en > avg_cs > avg_zh:
            eval_stats[lang(alt_set["orig"])][2] += 1
        if avg_en > avg_zh > avg_cs:
            eval_stats[lang(alt_set["orig"])][3] += 1
        if avg_zh > avg_cs > avg_en:
            eval_stats[lang(alt_set["orig"])][4] += 1
        if avg_zh > avg_en > avg_cs:
            eval_stats[lang(alt_set["orig"])][5] += 1

    return eval_stats

print(get_ranks(dev_set, dev_scores))
print(get_ranks(test_set, test_scores))
