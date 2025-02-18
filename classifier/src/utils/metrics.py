# utils/metrics.py

def calculate_accuracy(gold_answers, predictions):
    """
    Copied from your snippet. Compares each gold vs. prediction exactly.
    Returns percentage * 100. E.g. 90.0 means 90% accuracy.
    """
    total_acc_score = 0
    for (gold_answer, prediction) in zip(gold_answers, predictions):
        if gold_answer == prediction:
            total_acc_score += 1

    if len(gold_answers) == 0:
        return 0.0
    final_acc_score = (total_acc_score / len(gold_answers)) * 100
    return final_acc_score

def calculate_accuracy_perClass(gold_answers, predictions):
    """
    From your snippet, specifically for A/B/C classification.
    """
    a_total_acc_score = 0
    b_total_acc_score = 0
    c_total_acc_score = 0

    a_gold_num = len([i for i in gold_answers if i == 'A'])
    b_gold_num = len([i for i in gold_answers if i == 'B'])
    c_gold_num = len([i for i in gold_answers if i == 'C'])

    a_pred_num = len([i for i in predictions if i == 'A'])
    b_pred_num = len([i for i in predictions if i == 'B'])
    c_pred_num = len([i for i in predictions if i == 'C'])

    for (gold_answer, prediction) in zip(gold_answers, predictions):
        if gold_answer == prediction == 'A':
            a_total_acc_score += 1
        elif gold_answer == prediction == 'B':
            b_total_acc_score += 1
        elif gold_answer == prediction == 'C':
            c_total_acc_score += 1

    a_final_acc_score = -1 if a_gold_num == 0 else (a_total_acc_score / a_gold_num) * 100
    b_final_acc_score = -1 if b_gold_num == 0 else (b_total_acc_score / b_gold_num) * 100
    c_final_acc_score = -1 if c_gold_num == 0 else (c_total_acc_score / c_gold_num) * 100

    dict_final = {
        'A (zero) acc': a_final_acc_score,
        'B (single) acc': b_final_acc_score,
        'C (multi) acc': c_final_acc_score,
        'A (zero) pred num': a_pred_num,
        'B (single) pred num': b_pred_num,
        'C (multi) pred num': c_pred_num,
        'A (zero) gold num': a_gold_num,
        'B (single) gold num': b_gold_num,
        'C (multi) gold num': c_gold_num
    }
    return dict_final
