#coding=utf-8
from math import log2

def precision_at_k(actual, predicted, k):
    """
    计算准确率（Precision@k）

    Parameters:
    - actual: 实际的项目列表
    - predicted: 预测的项目列表
    - k: Top k 推荐

    Returns:
    - Precision@k
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = actual_set.intersection(predicted_set)
    return len(intersection) / k if k != 0 else 0

def recall_at_k(actual, predicted, k):
    """
    计算召回率（Recall@k）

    Parameters:
    - actual: 实际的项目列表
    - predicted: 预测的项目列表
    - k: Top k 推荐

    Returns:
    - Recall@k
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = actual_set.intersection(predicted_set)
    return len(intersection) / len(actual_set) if len(actual_set) != 0 else 0

def ndcg_at_k(actual, predicted, k):
    """
    计算归一化折损累积（NDCG@k）

    Parameters:
    - actual: 实际的项目列表
    - predicted: 预测的项目列表
    - k: Top k 推荐

    Returns:
    - NDCG@k
    """
    dcg = sum(1 / (log2(i + 2)) if item in actual else 0 for i, item in enumerate(predicted[:k]))
    idcg = sum(1 / (log2(i + 2)) for i in range(min(k, len(actual))))
    return dcg / idcg if idcg != 0 else 0


def ranking_evaluation(actual_items, predicted_items, k):
    precision_k = precision_at_k(actual_items, predicted_items, k)
    recall_k = recall_at_k(actual_items, predicted_items, k)
    ndcg_k = ndcg_at_k(actual_items, predicted_items, k)
    return precision_k, recall_k, ndcg_k
