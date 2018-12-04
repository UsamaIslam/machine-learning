import pandas as pd
from collections import defaultdict
import numpy as np
import re


def preprocess_string(str_arg):
    cleaned_str = re.sub('[^a-z\s]+', ' ', str_arg, flags=re.IGNORECASE)  # every char except alphabets is replaced
    cleaned_str = re.sub('(\s+)', ' ', cleaned_str)  # multiple spaces are replaced by single space
    cleaned_str = cleaned_str.lower()  # converting the cleaned string to lower case

    return cleaned_str  # returning the preprocessed string


class NaiveBayes:

    def __init__(self, unique_classes):

        self.classes = unique_classes  # Constructor is simply passed with unique number of classes of the training set

    def addToBow(self, example, dict_index):

        if isinstance(example, np.ndarray): example = example[0]

        for token_word in example.split():  # for every word in preprocessed example

            self.bow_dicts[dict_index][token_word] += 1  # increment in its count

    def train(self, dataset, labels):

        self.examples = dataset
        self.labels = labels
        self.bow_dicts = np.array([defaultdict(lambda: 0) for index in range(self.classes.shape[0])])
        # only convert to numpy arrays if initially not passed as numpy arrays - else its a useless recomputation

        if not isinstance(self.examples, np.ndarray): self.examples = np.array(self.examples)
        if not isinstance(self.labels, np.ndarray): self.labels = np.array(self.labels)

        # constructing BoW for each category
        for cat_index, cat in enumerate(self.classes):
            all_cat_examples = self.examples[self.labels == cat]  # filter all examples of category == cat
            cleaned_examples = [preprocess_string(cat_example) for cat_example in all_cat_examples]

            cleaned_examples = pd.DataFrame(data=cleaned_examples)
            np.apply_along_axis(self.addToBow, 1, cleaned_examples, cat_index)
        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            prob_classes[cat_index] = np.sum(self.labels == cat) / float(self.labels.shape[0])
            count = list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index] = np.sum(
                np.array(list(self.bow_dicts[cat_index].values()))) + 1  # |v| is remaining to be added
            all_words += self.bow_dicts[cat_index].keys()

        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]

        denoms = np.array(
            [cat_word_counts[cat_index] + self.vocab_length + 1 for cat_index, cat in enumerate(self.classes)])

        self.cats_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], denoms[cat_index]) for cat_index, cat in
                          enumerate(self.classes)]
        self.cats_info = np.array(self.cats_info)

    def getExampleProb(self, test_example):
        likelihood_prob = np.zeros(self.classes.shape[0])  # to store probability w.r.t each class
        for cat_index, cat in enumerate(self.classes):

            for test_token in test_example.split():  # split the test example and get p of each test word
                test_token_counts = self.cats_info[cat_index][0].get(test_token, 0) + 1
                test_token_prob = test_token_counts / float(self.cats_info[cat_index][2])
                likelihood_prob[cat_index] += np.log(test_token_prob)

        post_prob = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            post_prob[cat_index] = likelihood_prob[cat_index] + np.log(self.cats_info[cat_index][1])

        return post_prob

    def test(self, test_set):
        predictions = []  # to store prediction of each test example
        for example in test_set:
            cleaned_example = preprocess_string(example)
            post_prob = self.getExampleProb(cleaned_example)  # get prob of this example for both classes
            predictions.append(self.classes[np.argmax(post_prob)])
        return np.array(predictions)
