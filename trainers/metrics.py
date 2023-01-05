import json
from typing import Any, Dict, List, Optional

from rapidfuzz.process import cdist
from scipy.optimize import linear_sum_assignment


def _update_intent_count(
    counts: Dict[str, Any],
    intent: str,
    count_type: str,
) -> Dict[str, Any]:
    if intent not in counts:
        counts[intent] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "keys": {},
        }
    
    counts[intent][count_type] += 1
    
    return counts


def _update_key_count(
    counts: Dict[str, Any],
    intent: str,
    key: str,
    count_type: str,
) -> Dict[str, Any]:
    if key not in counts[intent]["keys"]:
        counts[intent]["keys"][key] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    
    counts[intent]["keys"][key][count_type] += 1
    
    return counts


def _positives_negatives_comment(
    true: List[Dict[str, Any]],
    pred: Optional[List[Dict[str, Any]]],
    base_property: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    
    # if pred is None, that means that the model output could not be parsed as
    # valid JSON, therefore everything that was in the true output maps to a
    # false negative.
    if pred is None:
        for t in range(len(true)):
            intent = true[t][base_property]
            
            out = _update_intent_count(
                counts=out,
                intent=intent,
                count_type="fn"
            )
            
            for key in true[t]:
                out = _update_key_count(
                    counts=out,
                    intent=intent,
                    key=key,
                    count_type="fn"
                )
        
        return out
    
    # pred is valid JSON.
    else:
        # First, go through predicted dicts. For each one that is exactly in
        # the true list, everything maps to a true positive. After counting
        # these TP values, delete the true and predicted dicts (to avoid double
        # counting).
        for p in range(len(pred) - 1, -1, -1):
            if pred[p] in true:
                t = true.index(pred[p])
                intent = true[t][base_property]
                
                out = _update_intent_count(
                    counts=out,
                    intent=intent,
                    count_type="tp"
                )
                
                for key in true[t]:
                    out = _update_key_count(
                        counts=out,
                        intent=intent,
                        key=key,
                        count_type="tp"
                    )
                
                del true[t]
                del pred[p]
        
        # For the remaining true and predicted dicts, first do fuzzy matching
        # to find the appropriate counterpart to compare against.
        true_s = [json.dumps(d) for d in true]
        pred_s = [json.dumps(d) for d in pred]
        
        similarities = cdist(queries=pred_s, choices=true_s)
        rows, cols = linear_sum_assignment(similarities, maximize=True)
        
        # Go through the matches and count the true/false positives and false
        # negatives. At the dict level, each will be counted as a false
        # positive because the dicts don't match exactly. At the key level,
        # compare the values to decide.
        for k in range(len(rows)):
            t = cols[k]
            p = rows[k]
            
            intent = true[t][base_property]
            
            out = _update_intent_count(
                counts=out,
                intent=intent,
                count_type="fp"
            )
            
            # For each key in the predicted dict:
            # - if the key is in the corresponding true dict:
            #   - if the values match, then it's a true positive.
            #   - if not, then it's a false positive.
            # - if the key is not in the corresponding true dict, it's a false
            #   positive.
            for key in list(pred[p]):
                if key in true[t]:
                    if pred[p][key] == true[t][key]:
                        out = _update_key_count(
                            counts=out,
                            intent=intent,
                            key=key,
                            count_type="tp"
                        )
                    else:
                        out = _update_key_count(
                            counts=out,
                            intent=intent,
                            key=key,
                            count_type="fp"
                        )
                    
                    del pred[p][key]
                    del true[t][key]
                else:
                    out = _update_key_count(
                        counts=out,
                        intent=intent,
                        key=key,
                        count_type="fp"
                    )
            
            # Go through the unaddressed keys in the true dict and count the
            # false negatives.
            for key in list(true[t]):
                out = _update_key_count(
                    counts=out,
                    intent=intent,
                    key=key,
                    count_type="fn"
                )
        
        # If there were more predicted dicts than true ones (or vice versa), we
        # need to go through the left over ones and add on the false positives
        # / negatives.
        for t in range(len(true)):
            if t not in cols:
                intent = true[t][base_property]
                
                out = _update_intent_count(
                    counts=out,
                    intent=intent,
                    count_type="fn"
                )
                
                for key in true[t]:
                    out = _update_key_count(
                        counts=out,
                        intent=intent,
                        key=key,
                        count_type="fn"
                    )
        
        for p in range(len(pred)):
            if p not in rows:
                try:
                    intent = pred[p][base_property]
                except KeyError:
                    intent = ""
                
                out = _update_intent_count(
                    counts=out,
                    intent=intent,
                    count_type="fp"
                )
                
                for key in pred[p]:
                    out = _update_key_count(
                        counts=out,
                        intent=intent,
                        key=key,
                        count_type="fp"
                    )
    
        return out


def positives_negatives(
    true: List[List[Dict[str, Any]]],
    pred: List[Optional[List[Dict[str, Any]]]],
    base_properties: List[str]
) -> Dict[str, Any]:
    totals: Dict[str, Any] = {}
    
    for i in range(len(true)):
        totals_i = _positives_negatives_comment(
            true=true[i],
            pred=pred[i],
            base_property=base_properties[i],
        )
        
        for intent in totals_i:
            if intent in totals:
                totals[intent]["tp"] += totals_i[intent]["tp"]
                totals[intent]["fp"] += totals_i[intent]["fp"]
                totals[intent]["fn"] += totals_i[intent]["fn"]
                
                for key in totals_i[intent]["keys"]:
                    if key in totals[intent]["keys"]:
                        totals[intent]["keys"][key]["tp"] += (
                            totals_i[intent]["keys"][key]["tp"]
                        )
                        totals[intent]["keys"][key]["fp"] += (
                            totals_i[intent]["keys"][key]["fp"]
                        )
                        totals[intent]["keys"][key]["fp"] += (
                            totals_i[intent]["keys"][key]["fp"]
                        )
                    else:
                        totals[intent]["keys"][key] = (
                            totals_i[intent]["keys"][key]
                        )
            else:
                totals[intent] = totals_i[intent]
    
    return totals


def precision_recall_f1(
    pos_neg: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    tp_objects = sum([pos_neg[intent]["tp"] for intent in pos_neg])
    fp_objects = sum([pos_neg[intent]["fp"] for intent in pos_neg])
    fn_objects = sum([pos_neg[intent]["fn"] for intent in pos_neg])
    
    tp_values = sum([
        pos_neg[intent]["keys"][key]["tp"]
        for intent in pos_neg for key in pos_neg[intent]["keys"]
    ])
    fp_values = sum([
        pos_neg[intent]["keys"][key]["fp"]
        for intent in pos_neg for key in pos_neg[intent]["keys"]
    ])
    fn_values = sum([
        pos_neg[intent]["keys"][key]["fn"]
        for intent in pos_neg for key in pos_neg[intent]["keys"]
    ])
    
    p_objects = (
        tp_objects / (tp_objects + fp_objects)
        if (tp_objects + fp_objects) > 0 else 0
    )
    r_objects = (
        tp_objects / (tp_objects + fn_objects)
        if (tp_objects + fn_objects) > 0 else 0
    )
    f1_objects = (
        (2 * p_objects * r_objects) / (p_objects + r_objects)
        if (p_objects + r_objects) > 0 else 0
    )
    
    p_values = (
        tp_values / (tp_values + fp_values)
        if (tp_values + fp_values) > 0 else 0
    )
    r_values = (
        tp_values / (tp_values + fn_values)
        if (tp_values + fn_values) > 0 else 0
    )
    f1_values = (
        (2 * p_values * r_values) / (p_values + r_values)
        if (p_values + r_values) > 0 else 0
    )
    
    return {
        "objects": {
            "precision": p_objects,
            "recall": r_objects,
            "f1": f1_objects,
        },
        "values": {
            "precision": p_values,
            "recall": r_values,
            "f1": f1_values,
        },
    }
