import math
import sys
import pickle
from tree_node import TreeNode, LeafNode
from helper_functions import *

# Constants
number_of_stumps = 4
number_of_features = 10
goal_state_column = 10


def partition_data(data_rows, col):
    true_rows, false_rows = [], []
    for row in data_rows:
        if row[col]:
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def all_belong_to_same_goal(rows, goal_state_col):
    goal_state = rows[0][goal_state_col]
    for row in rows:
        if row[goal_state_col] != goal_state:
            return False
    return goal_state


def entropy_helper(rows, goal_state_col):
    if len(rows) == 0:
        return 0
    no_of_english = 0
    for row in rows:
        if row[goal_state_col] == "en":
            no_of_english += 1
    no_of_dutch = len(rows) - no_of_english
    probability_of_english = no_of_english / len(rows)
    probability_of_dutch = no_of_dutch / len(rows)
    if probability_of_english == 0:
        return abs(probability_of_dutch * math.log(probability_of_dutch, 2))
    if probability_of_dutch == 0:
        return abs(probability_of_english * math.log(probability_of_english, 2))
    return abs(probability_of_english * math.log(probability_of_english, 2)
               + probability_of_dutch * math.log(probability_of_dutch, 2))


def entropy(true_rows, false_rows, goal_state_col):
    probability_of_true_rows = len(true_rows) / (len(true_rows) + len(false_rows))
    probability_of_false_rows = len(false_rows) / (len(true_rows) + len(false_rows))
    tempX = entropy_helper(true_rows, goal_state_col)
    tempY = entropy_helper(false_rows, goal_state_col)
    return probability_of_true_rows * tempX + probability_of_false_rows * tempY


def goal_with_higher_number_rows(rows, goal_state):
    a_number = 0
    b_number = 0
    for row in rows:
        if row[goal_state] == "en":
            a_number += 1
        else:
            b_number += 1
    return LeafNode("en") if a_number > b_number else LeafNode("nl")


def feature_extraction(data):
    number_of_examples = len(data)
    english_common_words, dutch_common_words = get_words()
    english_suffix, dutch_suffix, english_prefix, dutch_prefix, dutch_letter_combinations = get_suffix_prefix()
    features = []
    for line in data:
        language, words = line.split("|")
        feature = [False] * 10
        feature.append(language)
        words_length = 0
        number_of_english_words = 0
        number_of_dutch_words = 0
        for word in words.split():
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
        if sys.argv[3] != "dt":
            feature.append(1 / number_of_examples)
        features.append(feature)
    return features


def get_min_entropy_col(features, cols, goal_state_column):
    entropy_list = [0] * (goal_state_column + 1)
    min_entropy = 1
    min_entropy_col = -1
    for col in cols:
        true_rows, false_rows = partition_data(features, col)
        entropy_list[col] = entropy(true_rows, false_rows, goal_state_column)
        if entropy_list[col] <= min_entropy:
            min_entropy = entropy_list[col]
            min_entropy_col = col
    return min_entropy_col


def decision_tree_helper(features, cols, goal_state_column, parent_features):
    # Base Case
    if not features:
        return goal_with_higher_number_rows(parent_features, goal_state_column)
    value = all_belong_to_same_goal(features, goal_state_column)
    if value:
        return LeafNode(value)

    if len(cols) == 0:
        return goal_with_higher_number_rows(features, goal_state_column)

    min_entropy_col = get_min_entropy_col(features, cols, goal_state_column)
    true_rows, false_rows = partition_data(features, min_entropy_col)
    node = TreeNode(min_entropy_col)
    cols.remove(min_entropy_col)
    node.left = decision_tree_helper(true_rows, cols[:], goal_state_column, features)
    node.right = decision_tree_helper(false_rows, cols[:], goal_state_column, features)
    return node


def error_rate(true_rows, false_rows, true_goal, false_goal, goal_state_column):
    total_error = 0
    true_col_corrects = []
    true_col_errors = []
    false_col_corrects = []
    false_col_errors = []

    for row in true_rows:
        if row[goal_state_column] != true_goal:
            total_error += row[goal_state_column + 1]
            true_col_errors.append(row)
        else:
            true_col_corrects.append(row)
    for row in false_rows:
        if row[goal_state_column] != false_goal:
            total_error += row[goal_state_column + 1]
            false_col_errors.append(row)
        else:
            false_col_corrects.append(row)
    return total_error, true_col_errors, true_col_corrects, false_col_errors, false_col_corrects


def adaboost_helper(features, cols, goal_state_column, K):
    hypos = []
    for i in range(K):
        min_entropy_col = get_min_entropy_col(features, cols, goal_state_column)
        node = TreeNode(min_entropy_col)
        true_rows, false_rows = partition_data(features, min_entropy_col)
        true_goal = goal_with_higher_number_rows(true_rows, goal_state_column).value
        false_goal = goal_with_higher_number_rows(false_rows, goal_state_column).value
        node.left = LeafNode(true_goal)
        node.right = LeafNode(false_goal)
        total_error, true_col_errors, true_col_corrects, false_col_errors, false_col_corrects = \
            error_rate(true_rows, false_rows, true_goal, false_goal, goal_state_column)
        significance = 0.5 * math.log((1 - total_error) / total_error)
        hypos.append(tuple([node, significance]))
        features = []
        for feature in true_col_errors:
            feature[goal_state_column + 1] = feature[goal_state_column + 1] * math.exp(significance)
            features.append(feature)
        for feature in false_col_errors:
            feature[goal_state_column + 1] = feature[goal_state_column + 1] * math.exp(significance)
            features.append(feature)
        for feature in true_col_corrects:
            feature[goal_state_column + 1] = feature[goal_state_column + 1] * math.exp(-significance)
            features.append(feature)
        for feature in false_col_corrects:
            feature[goal_state_column + 1] = feature[goal_state_column + 1] * math.exp(-significance)
            features.append(feature)
        total_weight = 0
        for feature in features:
            total_weight = total_weight + feature[goal_state_column + 1]
        normalising_factor = 1 / total_weight
        total_weight = 0
        for feature in features:
            feature[goal_state_column + 1] = feature[goal_state_column + 1] * normalising_factor
            total_weight = total_weight + feature[goal_state_column + 1]
        cols.remove(min_entropy_col)
    return hypos


def decision_tree(features, cols):
    root = decision_tree_helper(features, cols, goal_state_column, features)
    with open(sys.argv[2], "wb") as output:
        pickle.dump(root, output, pickle.HIGHEST_PROTOCOL)


def adaboost(features, cols):
    hypos = adaboost_helper(features, cols, goal_state_column, number_of_stumps)
    with open(sys.argv[2], "wb") as output:
        pickle.dump(hypos, output, pickle.HIGHEST_PROTOCOL)


def main():
    data = read_input_file(sys.argv[1])
    features = feature_extraction(data)
    cols = [i for i in range(number_of_features)]
    if sys.argv[3] == "dt":
        decision_tree(features, cols)
    elif sys.argv[3] == "ada":
        adaboost(features, cols)


if __name__ == "__main__":
    main()
