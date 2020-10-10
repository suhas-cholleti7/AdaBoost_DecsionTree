import sys
from tree_node import TreeNode, LeafNode
from helper_functions import *
import pickle


def feature_extraction(data):
    english_common_words, dutch_common_words = get_words()
    english_suffix, dutch_suffix, english_prefix, dutch_prefix, dutch_letter_combinations = get_suffix_prefix()
    features = []
    for line in data:
        feature = [False] * 10
        words_length = 0
        number_of_english_words = 0
        number_of_dutch_words = 0
        for word in line.split():
            is_english = False
            is_dutch = False
            word = word.lower()
            if 'q' in word:
                feature[0] = True
            if word in dutch_letter_combinations:
                feature[1] = True
                is_dutch = True
            if word in english_common_words:
                feature[2] = True
                is_english = True
            if word in dutch_common_words:
                feature[3] = True
                is_dutch = True
            for suffix in english_suffix:
                if word.endswith(suffix):
                    feature[4] = True
                    is_english = True
                    break
            for suffix in dutch_suffix:
                if word.endswith(suffix):
                    feature[5] = True
                    is_dutch = True
                    break
            for prefix in english_prefix:
                if word.startswith(prefix):
                    feature[6] = True
                    is_english = True
                    break
            for prefix in dutch_prefix:
                if word.startswith(prefix):
                    feature[7] = True
                    is_dutch = True
                    break
            if is_english:
                number_of_english_words += 1
            if is_dutch:
                number_of_dutch_words += 1
            words_length = words_length + len(word)
        if words_length / 15 > 9:
            feature[8] = True
        if number_of_english_words > number_of_dutch_words:
            feature[9] = True
        features.append(feature)
    return features


def predict_decision(feature, node):
    if type(node) == LeafNode:
        return node.value
    if feature[int(node.value)]:
        return predict_decision(feature, node.left)
    else:
        return predict_decision(feature, node.right)


def decision(node, features):
    for feature in features:
        print(predict_decision(feature, node))


def adaboost_predict(feature, hypos):
    total_significance = 0
    for node, significance in hypos:
        if feature[int(node.value)]:
            if node.left.value == "en":
                total_significance += significance
            else:
                total_significance -= significance
        else:
            if node.right.value == "en":
                total_significance += significance
            else:
                total_significance -= significance
    if total_significance > 0:
        return "en"
    else:
        return "nl"


def adaboost(hypos, features):
    for feature in features:
        print(adaboost_predict(feature, hypos))


def main():
    data = read_input_file(sys.argv[1])
    features = feature_extraction(data)
    with open(sys.argv[2], 'rb') as handle:
        hypo = pickle.load(handle)
    if type(hypo) == TreeNode:
        decision(hypo, features)
    else:
        adaboost(hypo, features)


if __name__ == "__main__":
    main()
