import numpy as np

def generalized_F_beta_score(x: float, y: float, beta: float) -> float:
    """
    Compute the generalized F-beta score for two given values, `x` and `y`.
    
    This function generalizes the traditional F-beta score by treating `x` as precision
    and `y` as recall. It evaluates the harmonic mean of `x` and `y`, weighted by `beta`.
    
    Parameters:
        x (float) : 
            The first quantity
        y (float) : 
            The second quantity
        beta (float, optional) : 
            The weight of recall relative to precision (default is 2).

    Returns:

        Generalized_F_beta_score (float):
            The computed generalized F-beta score. Returns NaN if the denominator is zero.
    
    Examples:
        >>> generalized_F_beta_score(0.8, 0.6, beta=2)
        0.631578947368421
        >>> generalized_F_beta_score(0, 0)
        nan
    """
    return (1 + beta**2) * y * x / (beta**2 * x + y) if beta**2 * x + y > 0 \
                                                     else np.nan

def F_beta_score(precision : float, recall : float, beta : float) -> float :
    """
    Compute the F-beta score given precision and recall.
    
    This function evaluates the harmonic mean of precision and recall, weighted by `beta`.
    
    Parameters:
        precision (float): 
            The precision value
        recall (float): 
            The recall value
        beta (float, optional): 
            The weight of recall relative to precision (default is 2).
    
    Returns:
        F_beta_score (float):
            The computed F-beta score. Returns NaN if the denominator is zero.
    
    Examples:
        >>> F_beta_score(0.8, 0.6, beta=2)
        0.631578947368421
        >>> F_beta_score(0, 0)
        nan
    """
    return generalized_F_beta_score(precision, recall, beta)

def error_beta_score(fpr : float, fnr : float, beta : float) -> float :
    """
    Compute the error F-beta score given false positive rate (FPR) and false negative rate (FNR).
    
    This function generalizes the traditional F-beta score by treating `fpr` as precision
    and `fnr` as recall. It evaluates the harmonic mean of `fpr` and `fnr`, weighted by `beta`.
    
    Parameters:
        fpr (float): 
            The false positive rate
        fnr (float): 
            The false negative rate
        beta (float, optional): 
            The weight of recall relative to precision (default is 2).
    
    Returns:

        Error_beta_score (float):
            The computed error F-beta score.  Returns NaN if the denominator is zero.
    
    Examples:
        >>> error_beta_score(0.2, 0.4, beta=2)
        0.2857142857142857
        >>> error_beta_score(0, 0)
        nan
    """
    return generalized_F_beta_score(fnr, fpr, beta)

def performance(negatives : set, predicted_negatives : set, positives : set = set()) -> dict :
    """
    Compute performance metrics for predicted negatives against actual negatives.
    
    This function evaluates the Jaccard index, precision, recall, and F-beta score
    for the given sets of negatives and predicted negatives.
    
    Parameters:
        negatives (set): 
            The set of actual negative samples.
        predicted_negatives (set): 
            The set of predicted negative samples.
    
    Returns:
        dict: 
            A dictionary containing the computed performance metrics:
            - 'jaccard' (float): The Jaccard index.
            - 'precision' (float): The precision value.
            - 'recall' (float): The recall value.
            - 'f_beta' (float): The F-beta score.
    
    Examples:
        >>> negatives = {1, 2, 3, 4, 5}
        >>> predicted_negatives = {4, 5, 6, 7}
        >>> performance(negatives, predicted_negatives)
        {'jaccard': 0.2857142857142857, 'precision': 0.4, 'recall': 0.5, 'f_beta': 0.4444444444444444}
    """

    # Prendiamo, solo per valutare la metrica, come predetti positivi tutti quegli archi che 
    # non sono stati predetti come archi negativi, essendo che le strategie non ci danno delle
    # indicazioni per quanto riguardano gli archi positivi
    predicted_positives = (positives | negatives) - predicted_negatives

    jaccard = len(predicted_negatives & negatives) / len(predicted_negatives | negatives)
    precision = len(predicted_negatives & negatives) / (len(predicted_negatives & negatives) + len(predicted_negatives - negatives)) if (len(predicted_negatives & negatives) + len(predicted_negatives - negatives)) != 0 else np.nan
    recall = len(predicted_negatives & negatives) / (len(predicted_negatives & negatives) + len(predicted_positives - positives)) if (len(predicted_negatives & negatives) + len(predicted_positives - positives)) != 0 else np.nan
    f_beta = F_beta_score(precision, recall)
    # print(f'Bho: {len(predicted_positives & positives)}, Predetti positivi: {predicted_positives}, Positivi: {positives}')
    specificity = len(predicted_positives & positives) / (len(predicted_positives & positives) + len(predicted_negatives - negatives)) if (len(predicted_positives & positives) + len(predicted_negatives - negatives)) != 0 else np.nan

    return {'jaccard': jaccard, 
             'precision': precision,
             'recall': recall,
             'f_beta': f_beta,
             'specificity' : specificity}