import numpy as np


def get_classes(rewards, candidates, classes):
    """
    :param rewards: array of shape (n, d), of which we want to find the classes that the points belong to.
    Must be a subset of candidates
    :param candidates: array of shape (m ,d), of which we have the class indexes.
    :param classes: array of shape (m), indicating the class index of each point in candidates
    :return: array of shape (n), indicating class index of each point in rewards
    """
    n = rewards.shape[0]
    m = candidates.shape[0]
    d = rewards.shape[1]
    reward_classes = []
    
    for i in range(n):
        rows, cols = np.where(candidates == rewards[i])
        class_found = False
        for row in rows:
            if np.allclose(rewards[i], candidates[row]):
                reward_classes.append(classes[row])
                class_found = True
                break
        if not class_found:
            raise Exception("Class not found for point {}".format(i))
    
    return reward_classes


def class_proportion(classes, num_classes):
    """
    :param classes: array of shape (n), indicating class index of each point in rewards. From 0-9
    :param num_classes: int
    :return: tuple of (proportions, imbalance_score). proportions is a k-simplex where k is number of classes,
    imbalance_score is float scalar indicating level of imbalance between classes
    """
    proportions = np.bincount(classes, minlength=num_classes) / len(classes)
    imbalance_score = 1/num_classes * np.dot(proportions, proportions)
    
    return proportions, imbalance_score
