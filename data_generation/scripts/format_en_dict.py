cmu_map = {"DH": "S IY",
           "IH": "IY",
           "OY": "UW AO",
           "TH": "S",
           "V": "W",
           "ZH": "X"}


def cmu_to_phoneme(cmu_pronounciations):
    cmu_pronounciations = ''.join(c for c in cmu_pronounciations if not c.isdigit())
    cmu_phonemes = cmu_pronounciations.split()
    converted = []
    for pronounciation in cmu_phonemes:
        if pronounciation in cmu_map:
            converted.append(cmu_map[pronounciation])
        else:
            converted.append(pronounciation)
    return converted


en_words = dict()

with open("../dictionaries/cmudict-0.7b", 'r') as cmu_dict_f:
    for line in cmu_dict_f:
        if line[:3] == ";;;":
            continue
        en_word, pronounciation = line.split("  ")
        en_words[en_word] = pronounciation
print("Done extracting words from CMU dictionary")

with open("../extracted/en_dict.txt", 'w') as en_dict_f:
    for en_word, pronounciation in en_words.items():
        pronounciation = " ".join(cmu_to_phoneme(pronounciation))
        en_dict_f.write(f'{en_word.strip()}\t{pronounciation.strip()}\n')
print("Done writing dictionary")
