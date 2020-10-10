def read_input_file(input_file):
    data = []
    with open(input_file, encoding="utf8") as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def read_file(file):
    words = []
    with open(file) as f:
        for line in f.readlines():
            words.append(line.strip())
    return words


def get_words():
    dutch_common_words = read_file("dutch_words.txt")
    english_common_words = read_file("english_words.txt")
    return english_common_words, dutch_common_words


def get_suffix_prefix():
    english_suffix = ["tion", "sion", "ial", "able", "ible", "ful", "acy", "ance", "ism",
                      "ity", "ness", "ship", "ish", "ive", "less", "ious", "ify"]
    dutch_suffix = ["ische", "thisch", "thie", "achtig", "aan", "iek", "ief", "ier",
                    "iet", "een", "ant"]
    dutch_prefix = ["be", "ge", "her", "on", "ont", "ver", "niet", "oer"]
    english_prefix = ["dis", "ir", "un", "im"]
    dutch_letter_combinations = ['aai', 'ae', 'ai', 'au', 'eeu', 'ei', 'eu', 'ie', 'ieu', 'ij', 'oe', 'ou', 'oi', 'ooi', 'oei', 'ui']
    return english_suffix, dutch_suffix, english_prefix, dutch_prefix, dutch_letter_combinations
