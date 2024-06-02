def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score.

    Parameters:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        accuracy (float): The accuracy score.
    """
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    total = len(y_true)
    accuracy = correct / total
    return accuracy


def precision_score(y_true, y_pred, average='binary'):
    """
    Calculate precision score.

    Parameters: y_true (array-like): Ground truth (correct) target values. y_pred (array-like): Estimated targets as
    returned by a classifier. average (string, optional): Type of averaging to perform for multiclass. Possible
    values are - 'binary': Only report results for the class specified by `pos_label`. This is applicable only for
    binary classification. - 'micro': Calculate metrics globally by counting the total true positives,
    false negatives and false positives. - 'macro': Calculate metrics for each label, and find their unweighted mean.
    This does not take class imbalance into account. - 'weighted': Calculate metrics for each label, and find their
    average weighted by support (the number of true instances for each label).

    Returns:
        precision (float): The precision score.
    """
    if average == 'binary':
        true_positives = sum((yt == yp) and (yt == y_pred[0]) for yt, yp in zip(y_true, y_pred))
        false_positives = sum((yt != yp) and (yp == y_pred[0]) for yt, yp in zip(y_true, y_pred))
        if true_positives + false_positives == 0:
            return 0
        else:
            precision = true_positives / (true_positives + false_positives)
            return precision
    elif average in ['micro', 'macro', 'weighted']:
        precisions = []
        for label in set(y_true):
            true_positives_label = sum((yt == yp) and (yt == label) for yt, yp in zip(y_true, y_pred))
            false_positives_label = sum((yt != yp) and (yp == label) for yt, yp in zip(y_true, y_pred))
            if true_positives_label + false_positives_label == 0:
                precisions.append(0)
            else:
                precisions.append(true_positives_label / (true_positives_label + false_positives_label))

        if average == 'micro':
            precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
        elif average == 'macro':
            precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
        elif average == 'weighted':
            support = {label: sum(1 for yt in y_true if yt == label) for label in set(y_true)}
            precision = sum(p * support[label] for label, p in zip(set(y_true), precisions)) / sum(
                support.values()) if sum(support.values()) > 0 else 0

        return precision
    else:
        raise ValueError("Invalid value for 'average'. Possible values are 'binary', 'micro', 'macro', or 'weighted'.")


def recall_score(y_true, y_pred, average='binary'):
    """
    Calculate recall score.

    Parameters: y_true (array-like): Ground truth (correct) target values. y_pred (array-like): Estimated targets as
    returned by a classifier. average (string, optional): Type of averaging to perform for multiclass. Possible
    values are - 'binary': Only report results for the class specified by `pos_label`. This is applicable only for
    binary classification. - 'micro': Calculate metrics globally by counting the total true positives,
    false negatives and false positives. - 'macro': Calculate metrics for each label, and find their unweighted mean.
    This does not take class imbalance into account. - 'weighted': Calculate metrics for each label, and find their
    average weighted by support (the number of true instances for each label).

    Returns:
        recall (float): The recall score.
    """
    if average == 'binary':
        true_positives = sum((yt == yp) and (yt == y_pred[0]) for yt, yp in zip(y_true, y_pred))
        false_negatives = sum((yt != yp) and (yt == y_pred[0]) for yt, yp in zip(y_true, y_pred))
        if true_positives + false_negatives == 0:
            return 0
        else:
            recall = true_positives / (true_positives + false_negatives)
            return recall
    elif average in ['micro', 'macro', 'weighted']:
        recalls = []
        for label in set(y_true):
            true_positives_label = sum((yt == yp) and (yt == label) for yt, yp in zip(y_true, y_pred))
            false_negatives_label = sum((yt != yp) and (yt == label) for yt, yp in zip(y_true, y_pred))
            if true_positives_label + false_negatives_label == 0:
                recalls.append(0)
            else:
                recalls.append(true_positives_label / (true_positives_label + false_negatives_label))

        if average == 'micro':
            recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
        elif average == 'macro':
            recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
        elif average == 'weighted':
            support = {label: sum(1 for yt in y_true if yt == label) for label in set(y_true)}
            recall = sum(r * support[label] for label, r in zip(set(y_true), recalls)) / sum(support.values()) if sum(
                support.values()) > 0 else 0

        return recall
    else:
        raise ValueError("Invalid value for 'average'. Possible values are 'binary', 'micro', 'macro', or 'weighted'.")


def f1score(y_true, y_pred, average='binary'):
    """
    Calculate F1 score.

    Parameters: y_true (array-like): Ground truth (correct) target values. y_pred (array-like): Estimated targets as
    returned by a classifier. average (string, optional): Type of averaging to perform. Possible values are -
    'binary': Only report results for the class specified by `pos_label`. This is applicable only for binary
    classification. - 'micro': Calculate metrics globally by counting the total true positives, false negatives and
    false positives. - 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take
    class imbalance into account. - 'weighted': Calculate metrics for each label, and find their average weighted by
    support (the number of true instances for each label).

    Returns:
        f1 (float): The F1 score.
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)

    if precision + recall == 0:
        return 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error (MSE).

    Parameters:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets.

    Returns:
        mse (float): The mean squared error.
    """
    squared_errors = [(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)]
    mse = sum(squared_errors) / len(y_true)
    return mse


def mean_absolute_error(y_true, y_pred):
    """
    Calculate mean absolute error (MAE).

    Parameters:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets.

    Returns:
        mae (float): The mean absolute error.
    """
    absolute_errors = [abs(yt - yp) for yt, yp in zip(y_true, y_pred)]
    mae = sum(absolute_errors) / len(y_true)
    return mae


def r2_score(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination).

    Parameters:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets.

    Returns:
        r2 (float): The R-squared score.
    """
    mean_y_true = sum(y_true) / len(y_true)
    total_sum_of_squares = sum((yt - mean_y_true) ** 2 for yt in y_true)
    residual_sum_of_squares = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    r2 = 1- (residual_sum_of_squares / total_sum_of_squares)
    return r2
