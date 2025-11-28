import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn
from tqdm.notebook import tqdm
from pytorch_metric_learning.losses import SupConLoss
#from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from allennlp.modules.conditional_random_field import ConditionalRandomFieldWeightLannoy
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    multilabel_confusion_matrix,
    hamming_loss,
    classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
import pandas as pd

def compute_average_segment_length(true_boundaries_per_doc, doc_lengths):
    """
    Compute the average segment length across all documents based on ground truth boundaries.
    
    Parameters:
      - true_boundaries_per_doc: sorted dict mapping document names to lists of ground truth boundary indices
      - doc_lengths: dict mapping document names to the total number of lines in the document
    
    Returns:
      - average segment length: float
    """
    total_length = 0
    total_segments = 0
    
    for doc, boundaries in true_boundaries_per_doc.items():
        length = doc_lengths.get(doc, 0)
        
        segments = []
        if boundaries:
            # first segment: from index 0 to the first boundary
            segments.append(boundaries[0])
            # intermediate segments
            for i in range(1, len(boundaries)):
                segments.append(boundaries[i] - boundaries[i - 1])
            # last segment: from the last boundary to the end of the document
            segments.append(length - boundaries[-1])
        else:
            # the whole document is one segment
            segments.append(length)
        
        total_length += sum(segments)
        total_segments += len(segments)
    
    return total_length / total_segments if total_segments > 0 else 0.0

def compute_window_metrics(pred_boundaries_per_doc, true_boundaries_per_doc, doc_lengths, window_size):
    """
    Compute global (micro-averaged) Pk and WindowDiff scores over multiple documents.

    Parameters:
      - pred_boundaries_per_doc: sorted dict mapping document names to lists of predicted boundary indices.
      - true_boundaries_per_doc: sorted dict mapping document names to lists of true boundary indices.
      - doc_lengths: dict mapping document names to total number of tokens.
      - window_size: integer window size used for the sliding-window metrics. Half of avg segment length is a common choice.

    Returns:
      A tuple: (global_pk, global_windowdiff)
    """
    # Ensure window_size is an integer
    window_size = int(window_size)
    
    def in_same_segment(i, j, boundaries):
        for b in boundaries:
            if i < b <= j:
                return False
        return True

    def count_boundaries_in_window(i, j, boundaries):
        return sum(1 for b in boundaries if i < b <= j)

    total_pk_errors = 0
    total_wd_errors = 0
    total_windows = 0

    for doc, true_boundaries in true_boundaries_per_doc.items():
        true_boundaries = true_boundaries
        predicted_boundaries = pred_boundaries_per_doc.get(doc, [])
        
        num_tokens = int(doc_lengths[doc])
        windows = num_tokens - window_size
        if windows <= 0:
            continue
        
        errors_pk = 0
        errors_wd = 0
        
        for i in range(windows):
            ref_same = in_same_segment(i, i + window_size, true_boundaries)
            pred_same = in_same_segment(i, i + window_size, predicted_boundaries)
            if ref_same != pred_same:
                errors_pk += 1
            
            count_ref = count_boundaries_in_window(i, i + window_size, true_boundaries)
            count_pred = count_boundaries_in_window(i, i + window_size, predicted_boundaries)
            if count_ref != count_pred:
                errors_wd += 1
        
        total_pk_errors += errors_pk
        total_wd_errors += errors_wd
        total_windows += windows

    global_pk = total_pk_errors / total_windows if total_windows > 0 else 0.0
    global_windowdiff = total_wd_errors / total_windows if total_windows > 0 else 0.0

    return global_pk, global_windowdiff

def compute_basic_metrics(pred_boundaries_per_doc, true_boundaries_per_doc, task='multilabel'):
    """
    Compute precision, recall, F2 score, and support over multiple documents.
    
    The labels are the boundaries themselves (i.e., the indices of lines),
    and it is assumed that pred_boundaries_per_doc is already separated by class.
    
    Returns:
        precision (float): Micro-averaged precision.
        recall (float): Micro-averaged recall.
        f2 (float): Micro-averaged F2 score (β = 2).
        support (int): Total number of true boundaries (i.e., support).
    """
    if task == 'multilabel':
        total_true_positives = 0
        total_predicted = 0
        total_actual = 0

        for doc, true_boundaries in true_boundaries_per_doc.items():
            predicted_boundaries = pred_boundaries_per_doc.get(doc, [])

            true_set = set(true_boundaries)
            pred_set = set(predicted_boundaries)

            true_positives = len(true_set & pred_set)
            total_true_positives += true_positives
            total_predicted += len(pred_set)
            total_actual += len(true_set)
        
        precision = total_true_positives / total_predicted if total_predicted > 0 else 0.0
        recall = total_true_positives / total_actual if total_actual > 0 else 0.0
        
        # Compute F2 score (with beta=2)
        beta = 2.0
        if (beta**2 * precision + recall) == 0:
            f2 = 0.0
        else:
            f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        
        support = total_actual
        
        return precision, recall, f2, support

def compute_boundaries_multilabel(preds, gts, doc_names, class_names):
    """
    A boundary is defined as the index of the first line and/or end line of a segment.
    If no 'Begin' is assigned but the label set changes, the first line of the segment is considered a boundary.
    
    Args:
        preds: sorted dictionary of predictions, key: (document_name, line), value: labels:list of str
        gts: sorted dictionary of ground truths, key: (document_name, line), value: labels:list of str
        doc_names: list of document names
        class_names: list of class names
    """
    
    pred_boundaries_per_class_per_doc = {}
    true_boundaries_per_class_per_doc = {}
    
    for class_name in class_names:
        pred_boundaries_per_doc = {}
        true_boundaries_per_doc = {}

        for doc_name in doc_names:
            pred_boundaries_per_doc[doc_name] = []
            true_boundaries_per_doc[doc_name] = []
            
            # process predictions
            isClassnameinPreviousLine = False
            pastLineLabels = set()
            doc_preds = [[key[1], labels] for key, labels in preds.items() if key[0] == doc_name]
            
            for line, labels in doc_preds:
                labels_set = set(labels)
                if class_name in labels_set:
                    if isClassnameinPreviousLine == False: # if the previous line dit not contain the label
                        pred_boundaries_per_doc[doc_name].append(line)
                    elif 'Begin' in labels_set: # if we find a 'Begin' label
                        labels_set.remove('Begin')
                        if labels_set.issubset(pastLineLabels): # if the new label set is contained in the previous line set
                            pred_boundaries_per_doc[doc_name].append(line)
                    isClassnameinPreviousLine = True
                else:
                    if isClassnameinPreviousLine == True:
                        pred_boundaries_per_doc[doc_name].append(line)
                    isClassnameinPreviousLine = False
                pastLineLabels = labels_set
            
            # process ground truths
            isClassnameinPreviousLine = False
            pastLineLabels = set()
            doc_gts = [[key[1], labels] for key, labels in gts.items() if key[0] == doc_name]
            
            for line, labels in doc_gts:
                labels_set = set(labels)
                if class_name in labels_set: # if the previous line contains the label
                    if isClassnameinPreviousLine == False: # if the previous line dit not contain the label
                        true_boundaries_per_doc[doc_name].append(line)
                    elif 'Begin' in labels_set: # if we find a 'Begin' label
                        labels_set.remove('Begin')
                        if labels_set.issubset(pastLineLabels): # if the new label set is contained in the previous line set
                            true_boundaries_per_doc[doc_name].append(line)
                    isClassnameinPreviousLine = True
                else: # if the previous line does not contain the label
                    if isClassnameinPreviousLine == True: # and the current line does
                        true_boundaries_per_doc[doc_name].append(line)
                    isClassnameinPreviousLine = False
                pastLineLabels = labels_set

            pred_boundaries_per_class_per_doc[class_name] = pred_boundaries_per_doc
            true_boundaries_per_class_per_doc[class_name] = true_boundaries_per_doc
        
    return pred_boundaries_per_class_per_doc, true_boundaries_per_class_per_doc

def compute_boundaries_multiclass(preds, gts, doc_names):
    """
    A boundary is defined as the index of the first line of a segment (excluding the first segment).
    For simple labels, a boundary occurs when the label changes from the previous line.
    
    Args:
        preds: sorted dictionary of predictions, key: (document_name, line), value: label
        gts: sorted dictionary of ground truths, key: (document_name, line), value: label
        doc_names: list of document names
    
    Returns:
        pred_boundaries_per_doc: dictionary with the predicted boundaries per document
        true_boundaries_per_doc: dictionary with the true boundaries per document
    """
    
    pred_boundaries_per_doc = {}
    true_boundaries_per_doc = {}
    
    for doc in doc_names:
        pred_boundaries_per_doc[doc] = []
        true_boundaries_per_doc[doc] = []
        
        # Process predictions: a boundary is when the current label differs from the previous one
        prev_label = None
        for key, label in preds.items():
            document_name, line = key
            if document_name == doc:
                if prev_label is not None and label != prev_label:
                    pred_boundaries_per_doc[doc].append(line)
                prev_label = label
        
        # Process ground truths similarly
        prev_label = None
        for key, label in gts.items():
            document_name, line = key
            if document_name == doc:
                if prev_label is not None and label != prev_label:
                    true_boundaries_per_doc[doc].append(line)
                prev_label = label
                
    return pred_boundaries_per_doc, true_boundaries_per_doc

def compute_boundaries_BIO(preds, gts, doc_names):
    """
    A boundary is defined as the index of the first line of a segment (excluding the first segment),
    i.g., a B-label line index or the first line index of an 'O' sequence.
    
    args:
        preds: sorted dictionary of predictions, key: (document_name, line), value: label
        gts: sorted dictionary of ground truths, key: (document_name, line), value: label
        doc_names: list of document names
    
    returns:
        pred_boundaries_per_doc: dictionary with the predicted boundaries per document
        true_boundaries_per_doc: dictionary with the true boundaries per document
    """
    
    pred_boundaries_per_doc = {}
    true_boundaries_per_doc = {}
    
    for doc in doc_names:
        pred_boundaries_per_doc[doc] = []
        true_boundaries_per_doc[doc] = []
        
        # find boundaries in the predictions
        past_boundary = None
        for key, label in preds.items():
            document_name, line = key
            if document_name == doc:
                if label.startswith("B"): # B-label
                    pred_boundaries_per_doc[document_name].append(line)
                    past_boundary = label
                elif label == 'O' and past_boundary != 'O': # first token of an 'O' sequence
                    pred_boundaries_per_doc[document_name].append(line)
                    past_boundary = label
                elif label.startswith("I") and past_boundary != None and past_boundary != 'O': # I-label and the previous boundary was a B or I label from another class
                    if (label[2:] != past_boundary[2:]):
                        pred_boundaries_per_doc[document_name].append(line)
                        past_boundary = label
                elif label.startswith("I") and (past_boundary == None or past_boundary == 'O'): # I-label and the previous boundary was an O-label or None
                    pred_boundaries_per_doc[document_name].append(line)
                    past_boundary = label
                    
        if pred_boundaries_per_doc[doc]: # remove the first boundary
            pred_boundaries_per_doc[doc] = pred_boundaries_per_doc[doc][1:]
        
        # find boundaries in the ground truth labels
        past_boundary = None
        for key, label in gts.items():
            document_name, line = key
            if document_name == doc:
                if label.startswith("B"): # B-label
                    true_boundaries_per_doc[document_name].append(line)
                    past_boundary = label
                elif label == 'O' and past_boundary != 'O': # first token of an 'O' sequence
                    true_boundaries_per_doc[document_name].append(line)
                    past_boundary = label
                elif label.startswith("I") and past_boundary != None and past_boundary != 'O': # I-label and the previous boundary was a B or I label from another class
                    if (label[2:] != past_boundary[2:]):
                        true_boundaries_per_doc[document_name].append(line)
                        past_boundary = label
                elif label.startswith("I") and (past_boundary == None or past_boundary == 'O'): # I-label and the previous boundary was an O-label or None
                    true_boundaries_per_doc[document_name].append(line)
                    past_boundary = label
                    
        if true_boundaries_per_doc[doc]: # remove the first boundary
            true_boundaries_per_doc[doc] = true_boundaries_per_doc[doc][1:]
            
    return pred_boundaries_per_doc, true_boundaries_per_doc

def post_process(preds, doc_names, task, exhibit_code):
    #if any 'Exhibit' label with a given int label is predicted, extend the prediction to the whole document
    if task == 'multilabel':
        for doc_name in doc_names:
            detected_exhibit = False
            for key, labels in preds.items():
                if key[0] == doc_name:
                    if labels[exhibit_code] == 1:
                        detected_exhibit = True
                    elif detected_exhibit:
                        labels[exhibit_code] = 1
    return preds
    
def evaluate_model(model, dataloader, labels_dict, save_path, task, device='cpu'):
    
    # get predictions and assign them to their ids
    model.eval()

    all_preds_with_ids = []
    all_labels_with_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            input_embeddings = batch['input_embeddings'].to(device)
            sentence_ids = batch['sentence_ids']
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            if task == 'multilabel':
                preds = model(input_embeddings, attention_mask)
            else:
                preds, swap_logits = model(input_embeddings, attention_mask=attention_mask)
                
            mask = attention_mask.to('cpu').numpy().astype(bool)
            
            preds_with_ids = []
            labels_with_ids = []
            for i in range(preds.shape[0]):
                for j in range(preds.shape[1]):
                    if mask[i][j]:
                        if task == 'multilabel':
                            pred_vector = preds[i][j].detach().cpu().tolist()
                            label_vector = labels[i][j].detach().cpu().tolist()
                            all_preds_with_ids.append([sentence_ids[i][j][0], sentence_ids[i][j][1], pred_vector])
                            all_labels_with_ids.append([sentence_ids[i][j][0], sentence_ids[i][j][1], label_vector])
                        else:
                            preds_with_ids.append([sentence_ids[i][j][0], sentence_ids[i][j][1], preds[i][j].item()])
                            labels_with_ids.append([sentence_ids[i][j][0], sentence_ids[i][j][1], labels[i][j].item()])
            
            all_preds_with_ids.extend(preds_with_ids)
            all_labels_with_ids.extend(labels_with_ids)

    # keep only the predictions of the first pass
    all_preds_with_ids_dict = {}
    all_labels_with_ids_dict = {}
    for i, item in enumerate(all_preds_with_ids):
        key = (item[0], item[1])
        if key not in all_preds_with_ids_dict:
            all_preds_with_ids_dict[key] = item[2]
            all_labels_with_ids_dict[key] = all_labels_with_ids[i][2]

    # sort preds and labels for saving        
    all_preds_with_ids_dict = {k: v for k, v in sorted(all_preds_with_ids_dict.items())}
    all_labels_with_ids_dict = {k: v for k, v in sorted(all_labels_with_ids_dict.items())}

    # save the predictions with sentences for each document to json files
    doc_names = set([item[0][0] for item in all_preds_with_ids_dict.items()])
    
    if task == 'multilabel':
        # apply post-processing to predictions
        exhibit_code = labels_dict['Exhibit']
        all_preds_with_ids_dict = post_process(all_preds_with_ids_dict, doc_names, task, exhibit_code)
        
        # convert label multi-hot vectors to textual classes
        code2label = {v: k for k, v in labels_dict.items()}
        all_preds_textual = {k: [code2label[i] for i, v in enumerate(v) if v == 1] for k, v in all_preds_with_ids_dict.items()}
        all_labels_textual = {k: [code2label[i] for i, v in enumerate(v) if v == 1] for k, v in all_labels_with_ids_dict.items()}
        # map to 'O' if no label is predicted
        for key in all_preds_textual.keys():
            if not all_preds_textual[key]:
                all_preds_textual[key] = ['O']
        for key in all_labels_textual.keys():
            if not all_labels_textual[key]:
                all_labels_textual[key] = ['O']
    else:
        # convert label codes to textual classes
        code2label = {v: k for k, v in labels_dict.items()}
        all_preds_textual = {k: code2label[v] for k, v in all_preds_with_ids_dict.items()}
        all_labels_textual = {k: code2label[v] for k, v in all_labels_with_ids_dict.items()}

    for doc_name in doc_names:
        doc_results = []
        with open(doc_name, "r", encoding="utf-8-sig", errors="ignore") as file:
                doc_content = json.load(file)
        for key, prediction in all_preds_textual.items():
            if key[0] == doc_name:
                sentence = doc_content[key[1]]["sentence"]
                if task == 'multilabel':
                    doc_results.append({"sentence": sentence, "predicted_labels": prediction})
                else:
                    doc_results.append({"sentence": sentence, "predicted_label": prediction})
        json.dump(doc_results, open(save_path + f"/{doc_name.split('/')[-1]}", "w"), indent=4, ensure_ascii=False)

    ################################# Classification metrics #################################    
    all_preds = np.array(list(all_preds_with_ids_dict.values()))
    all_labels = np.array(list(all_labels_with_ids_dict.values()))

    if task == 'multilabel':
        class_names = [k for k, v in labels_dict.items()]
    else:
        unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
        class_names = [k for k, v in labels_dict.items() if v in unique_labels]

    accuracy = accuracy_score(all_labels, all_preds)
    micro_precision, micro_recall, micro_f2, _ = precision_recall_fscore_support(all_labels, all_preds, beta=2, average='micro', zero_division=0)
    macro_precision, macro_recall, macro_f2, _ = precision_recall_fscore_support(all_labels, all_preds, beta=2, average='macro', zero_division=0)
    weighted_precision, weighted_recall, weighted_f2, _ = precision_recall_fscore_support(all_labels, all_preds, beta=2, average='weighted', zero_division=0)
    
    print("\nGlobal classification metrics\n")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F2 Score: {micro_f2:.4f}")
    print("-----------------")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F2 Score: {macro_f2:.4f}")
    print("-----------------")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F2 Score: {weighted_f2:.4f}")

    if task == 'multilabel':
        hamming_loss_score = hamming_loss(all_labels, all_preds)
        print("-----------------")
        print(f"Exact match ratio: {accuracy:.4f}")
        print(f"Hamming Loss (lower is better): {hamming_loss_score:.4f}")
        print("-----------------")
        
        # multilabel confusion matrix
        mcm = multilabel_confusion_matrix(all_labels, all_preds)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Multi-label Classification Confusion Matrices')
        axes = axes.flatten()

        for i, cm in enumerate(mcm):
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(class_names[i])

        fig.tight_layout()
        fig.show()
        
        # multilabel classification report
        class_report = classification_report(all_labels, all_preds, zero_division=0, target_names=class_names)
        print(class_report)
        
    else:
        print("-----------------")
        print(f"Accuracy: {accuracy:.4f}")
        print("-----------------")
        
        # confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true', labels=unique_labels)
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matrix, annot=False, fmt='.1f', cmap='viridis', xticklabels=class_names, yticklabels=class_names, linewidths=.5)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix, normalized by row')
        plt.show()
        
        # classification report
        class_report = classification_report(all_labels, all_preds, zero_division=0, labels = unique_labels, target_names=class_names)
        print(class_report)

    
    ################################## Segmentation metrics ##################################    
    if task == 'multilabel':
        pred_boundaries_per_class_per_doc, true_boundaries_per_class_per_doc = compute_boundaries_multilabel(all_preds_textual, all_labels_textual, doc_names, class_names)
        doc_lens = {}
        for doc in doc_names:
            with open(doc, "r", encoding="utf-8-sig", errors="ignore") as file:
                doc_lens[doc] = len(json.load(file))
        
        metrics_per_class = {}
        global_segmentation_metrics = {'macro': {'precision': 0, 'recall': 0, 'f2': 0, 'pk': 0, 'windowdiff': 0},
                                       'weighted': {'precision': 0, 'recall': 0, 'f2': 0, 'pk': 0, 'windowdiff': 0}}
        sum_support = 0
        
        for class_name in class_names:
            precision, recall, f2, support = compute_basic_metrics(pred_boundaries_per_class_per_doc[class_name], true_boundaries_per_class_per_doc[class_name])
            avg_segment_len = compute_average_segment_length(true_boundaries_per_class_per_doc[class_name], doc_lens)
            pk, windowdiff = compute_window_metrics(pred_boundaries_per_class_per_doc[class_name], true_boundaries_per_class_per_doc[class_name], doc_lens, avg_segment_len/2)
            metrics_per_class[class_name] = {"Precision": precision, "Recall": recall, "F2 Score": f2, "1 - Pk": 1 - pk, "1 - Windowdiff": 1 - windowdiff}
            
            global_segmentation_metrics["macro"]["precision"] += precision
            global_segmentation_metrics["macro"]["recall"] += recall
            global_segmentation_metrics["macro"]["f2"] += f2
            global_segmentation_metrics["macro"]["pk"] += pk
            global_segmentation_metrics["macro"]["windowdiff"] += windowdiff
            
            global_segmentation_metrics["weighted"]["precision"] += precision * support
            global_segmentation_metrics["weighted"]["recall"] += recall * support
            global_segmentation_metrics["weighted"]["f2"] += f2 * support
            global_segmentation_metrics["weighted"]["pk"] += pk * support
            global_segmentation_metrics["weighted"]["windowdiff"] += windowdiff * support
            sum_support += support
            
        for metric in global_segmentation_metrics["macro"].keys():
            global_segmentation_metrics["macro"][metric] /= len(class_names)
        for metric in global_segmentation_metrics["weighted"].keys():
            global_segmentation_metrics["weighted"][metric] /= sum_support
            
        # plot metrics per class
        df = pd.DataFrame(metrics_per_class).T

        # Create a grouped bar chart
        ax = df.plot(kind='bar', figsize=(12, 6))
        ax.set_title("Segmentation metrics per class")
        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(title="Metrics")
        plt.tight_layout()
        plt.show()  
        
        # print global metrics
        print("\nGlobal segmentation metrics\n")
        print(f"Macro Precision: {global_segmentation_metrics['macro']['precision']:.4f}")
        print(f"Macro Recall: {global_segmentation_metrics['macro']['recall']:.4f}")
        print(f"Macro F2 Score: {global_segmentation_metrics['macro']['f2']:.4f}")
        print(f"Macro Pk Score: {global_segmentation_metrics['macro']['pk']:.4f}")
        print(f"Macro WindowDiff: {global_segmentation_metrics['macro']['windowdiff']:.4f}")
        print("-----------------")
        print(f"Weighted Precision: {global_segmentation_metrics['weighted']['precision']:.4f}")
        print(f"Weighted Recall: {global_segmentation_metrics['weighted']['recall']:.4f}")
        print(f"Weighted F2 Score: {global_segmentation_metrics['weighted']['f2']:.4f}")
        print(f"Weighted Pk Score: {global_segmentation_metrics['weighted']['pk']:.4f}")
        print(f"Weighted WindowDiff: {global_segmentation_metrics['weighted']['windowdiff']:.4f}")
        
    else:
        if task == "BIO":
            pred_boundaries_per_doc, true_boundaries_per_doc = compute_boundaries_BIO(all_preds_textual, all_labels_textual, doc_names)
        elif task == "multiclass":
            pred_boundaries_per_doc, true_boundaries_per_doc = compute_boundaries_multiclass(all_preds_textual, all_labels_textual, doc_names)
        
        # compute global micro metrics which are strict to boundary position
        micro_precision, micro_recall, micro_f1 = compute_basic_metrics(pred_boundaries_per_doc, true_boundaries_per_doc)
        
        # compute global pk-score and windowdiff metrics which are less strict to boundary position
        doc_lens = {}
        for doc in doc_names:
            with open(doc, "r", encoding="utf-8-sig", errors="ignore") as file:
                doc_lens[doc] = len(json.load(file))
        avg_segment_len = compute_average_segment_length(true_boundaries_per_doc, doc_lens)
        global_pk, global_windowdiff, _ = compute_window_metrics(pred_boundaries_per_doc, true_boundaries_per_doc, doc_lens, avg_segment_len/2)
        
        print("\nSegmentation results\n")
        print(f"Micro Precision: {micro_precision:.4f}")
        print(f"Micro Recall: {micro_recall:.4f}")
        print(f"Micro F1 Score: {micro_f1:.4f}")
        print(f"Global Pk Score (micro avg): {global_pk:.4f}")
        print(f"Global WindowDiff (micro avg): {global_windowdiff:.4f}")

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    hamming_loss,
    multilabel_confusion_matrix,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

# Assume the following segmentation functions are available:
# from your_segmentation_module import (compute_boundaries_multilabel, compute_boundaries_BIO, 
#     compute_boundaries_multiclass, compute_basic_metrics, compute_average_segment_length, compute_window_metrics)

def evaluate_from_json_folder(gt_folder, pred_folder, labels_dict, save_path, task):
    """
    Evaluate predictions against ground truth JSON files from folders.
    
    Both gt_folder and pred_folder should contain JSON files. Each JSON file is a list of dictionaries 
    with "sentence" and "labels" keys. The file names in both folders must match.
    
    The function processes each contract, saves per-contract evaluation files in save_path, and computes 
    global classification and segmentation metrics.
    
    Args:
        gt_folder (str): Folder containing ground-truth JSON files.
        pred_folder (str): Folder containing prediction JSON files.
        labels_dict (dict): Mapping from label names (str) to codes (int).
        save_path (str): Directory where per-document evaluation files will be saved.
        task (str): Either 'multilabel' or another type (e.g., "BIO" or "multiclass") for single-label tasks.
    """
    os.makedirs(save_path, exist_ok=True)
    
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith('.json')]
    if not gt_files:
        print("No JSON files found in the ground truth folder!")
        return

    # Global containers for classification metrics and segmentation metrics.
    # For segmentation, we build dictionaries keyed by (doc, sentence_index)
    all_preds_textual = {}
    all_labels_textual = {}
    
    # For classification metrics:
    classification_gt = []
    classification_pred = []
    
    # For multilabel, we need the full sorted set of class names.
    if task == 'multilabel':
        all_label_names = sorted(labels_dict.keys())
    
    # Set to collect document paths (full path to ground truth files)
    doc_names = set()
    
    # Process each contract file
    for filename in tqdm(gt_files, desc="Processing files"):
        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        
        if not os.path.exists(pred_path):
            #print(f"Prediction file for {filename} not found, skipping.")
            continue
        
        # Load the ground truth and prediction JSON for this contract.
        with open(gt_path, 'r', encoding="utf-8-sig", errors="ignore") as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding="utf-8-sig", errors="ignore") as f:
            pred_data = json.load(f)
        
        if len(gt_data) != len(pred_data):
            print(f"Warning: In file {filename}, the number of sentences in ground truth and prediction do not match!")
        
        doc_names.add(gt_path)
        doc_results = []  # For saving per-document evaluation
        
        # Process each sentence (assume order is aligned)
        for idx, (gt_entry, pred_entry) in enumerate(zip(gt_data, pred_data)):
            # Get textual labels; if empty, default to ["O"]
            gt_labels = gt_entry.get("labels", []) or ["O"]
            pred_labels = pred_entry.get("labels", []) or ["O"]
            
            # Store in segmentation dictionaries using key: (doc, sentence index)
            all_labels_textual[(gt_path, idx)] = gt_labels
            all_preds_textual[(gt_path, idx)] = pred_labels
            
            # For classification metrics, build indicator vectors (for multilabel) or use first label (for single-label)
            if task == 'multilabel':
                # Build binary vectors for all classes
                gt_vector = [1 if label in gt_labels else 0 for label in all_label_names]
                pred_vector = [1 if label in pred_labels else 0 for label in all_label_names]
                classification_gt.append(gt_vector)
                classification_pred.append(pred_vector)
            else:
                gt_label = gt_labels[0]
                pred_label = pred_labels[0]
                classification_gt.append(gt_label)
                classification_pred.append(pred_label)
            
            # Build per-document results (use the ground truth sentence for reference)
            doc_results.append({
                "sentence": gt_entry.get("sentence", ""),
                "ground_truth_labels": gt_labels,
                "predicted_labels": pred_labels
            })
        
        # Save per-document evaluation results
        with open(os.path.join(save_path, filename), "w", encoding="utf-8-sig") as out_file:
            json.dump(doc_results, out_file, indent=4, ensure_ascii=False)
    
    ################################# Global Classification Metrics #################################
    print('-' * 50 + ' Global Classification Metrics ' + '-' * 50)
    
    if task == 'multilabel':
        gt_arr = np.array(classification_gt)
        pred_arr = np.array(classification_pred)
        
        accuracy = accuracy_score(gt_arr, pred_arr)
        micro_precision, micro_recall, micro_f2, _ = precision_recall_fscore_support(gt_arr, pred_arr, beta=2, average='micro', zero_division=0)
        macro_precision, macro_recall, macro_f2, _ = precision_recall_fscore_support(gt_arr, pred_arr, beta=2, average='macro', zero_division=0)
        weighted_precision, weighted_recall, weighted_f2, _ = precision_recall_fscore_support(gt_arr, pred_arr, beta=2, average='weighted', zero_division=0)
        
        print("\nGlobal classification metrics\n")
        print(f"Micro Precision: {micro_precision:.4f}")
        print(f"Micro Recall: {micro_recall:.4f}")
        print(f"Micro F2 Score: {micro_f2:.4f}")
        print("-----------------")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F2 Score: {macro_f2:.4f}")
        print("-----------------")
        print(f"Weighted Precision: {weighted_precision:.4f}")
        print(f"Weighted Recall: {weighted_recall:.4f}")
        print(f"Weighted F2 Score: {weighted_f2:.4f}")

        hamming_loss_score = hamming_loss(gt_arr, pred_arr)
        print("-----------------")
        print(f"Exact match ratio: {accuracy:.4f}")
        print(f"Hamming Loss (lower is better): {hamming_loss_score:.4f}")
        print("-----------------")


        
        # Plot multilabel confusion matrices per class
        mcm = multilabel_confusion_matrix(gt_arr, pred_arr)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Multi-label Classification Confusion Matrices')
        axes = axes.flatten()
        for i, cm in enumerate(mcm):
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(all_label_names[i])
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        fig.tight_layout()
        fig.show()

        class_report = classification_report(gt_arr, pred_arr, zero_division=0, target_names=all_label_names)
        print(class_report)

        """
        num_classes = len(all_label_names)
        ncols = min(num_classes, 5)
        nrows = (num_classes + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = axes.flatten() if num_classes > 1 else [axes]
        for i, cm in enumerate(mcm):
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(all_label_names[i])
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        fig.tight_layout()
        plt.show()
        """
    else:
        accuracy = accuracy_score(classification_gt, classification_pred)
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            classification_gt, classification_pred, average='micro', zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            classification_gt, classification_pred, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            classification_gt, classification_pred, average='weighted', zero_division=0
        )
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Micro Precision: {micro_precision:.4f}")
        print(f"Micro Recall: {micro_recall:.4f}")
        print(f"Micro F1 Score: {micro_f1:.4f}")
        print("-" * 25)
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print("-" * 25)
        print(f"Weighted Precision: {weighted_precision:.4f}")
        print(f"Weighted Recall: {weighted_recall:.4f}")
        print(f"Weighted F1 Score: {weighted_f1:.4f}")
        
        unique_labels = np.unique(classification_gt + classification_pred)
        conf_matrix = confusion_matrix(classification_gt, classification_pred, normalize='true', labels=unique_labels)
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='viridis',
                    xticklabels=unique_labels, yticklabels=unique_labels, linewidths=.5)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Normalized by Row)')
        plt.show()
        
        class_report = classification_report(classification_gt, classification_pred, zero_division=0, labels=unique_labels)
        print(class_report)
    
    ################################## Segmentation Metrics ##################################
    print('\n' + '-' * 50 + ' Segmentation Metrics ' + '-' * 50)
    
    # For segmentation metrics, we need to compute boundaries.
    # We reuse the dictionaries all_preds_textual and all_labels_textual built earlier.
    # Also, compute document lengths (number of sentences per doc) by reading the ground-truth files.
    doc_lens = {}
    for doc in doc_names:
        try:
            with open(doc, "r", encoding="utf-8-sig", errors="ignore") as f:
                doc_content = json.load(f)
            doc_lens[doc] = len(doc_content)
        except Exception as e:
            print(f"Error reading {doc} for segmentation metrics: {e}")
            doc_lens[doc] = 0

    if task == 'multilabel':
        # Compute boundaries per class per document.
        # all_label_names is the list of class labels.
        pred_boundaries_per_class_per_doc, true_boundaries_per_class_per_doc = \
            compute_boundaries_multilabel(all_preds_textual, all_labels_textual, doc_names, all_label_names)
        
        metrics_per_class = {}
        global_segmentation_metrics = {'macro': {'precision': 0, 'recall': 0, 'f2': 0, 'pk': 0, 'windowdiff': 0},
                                       'weighted': {'precision': 0, 'recall': 0, 'f2': 0, 'pk': 0, 'windowdiff': 0}}
        sum_support = 0

        all_label_names = [class_name for class_name in all_label_names if class_name != 'Begin']
        for class_name in all_label_names:
            precision, recall, f2, support = compute_basic_metrics(pred_boundaries_per_class_per_doc[class_name], true_boundaries_per_class_per_doc[class_name])
            avg_segment_len = compute_average_segment_length(true_boundaries_per_class_per_doc[class_name], doc_lens)
            pk, windowdiff = compute_window_metrics(pred_boundaries_per_class_per_doc[class_name], true_boundaries_per_class_per_doc[class_name], doc_lens, avg_segment_len/2)
            metrics_per_class[class_name] = {"Precision": precision, "Recall": recall, "F2 Score": f2, "1 - Pk": 1 - pk, "1 - Windowdiff": 1 - windowdiff}
            
            global_segmentation_metrics["macro"]["precision"] += precision
            global_segmentation_metrics["macro"]["recall"] += recall
            global_segmentation_metrics["macro"]["f2"] += f2
            global_segmentation_metrics["macro"]["pk"] += pk
            global_segmentation_metrics["macro"]["windowdiff"] += windowdiff
            
            global_segmentation_metrics["weighted"]["precision"] += precision * support
            global_segmentation_metrics["weighted"]["recall"] += recall * support
            global_segmentation_metrics["weighted"]["f2"] += f2 * support
            global_segmentation_metrics["weighted"]["pk"] += pk * support
            global_segmentation_metrics["weighted"]["windowdiff"] += windowdiff * support
            sum_support += support

        for metric in global_segmentation_metrics["macro"].keys():
            global_segmentation_metrics["macro"][metric] /= len(all_label_names)
        for metric in global_segmentation_metrics["weighted"].keys():
            global_segmentation_metrics["weighted"][metric] /= sum_support
        
        # plot metrics per class
        df = pd.DataFrame(metrics_per_class).T

        # Create a grouped bar chart
        ax = df.plot(kind='bar', figsize=(12, 6))

        ax.set_title("Segmentation metrics per class")
        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(title="Metrics")
        plt.tight_layout()
        plt.show() 

        # print global metrics
        print("\nGlobal segmentation metrics\n")
        print(f"Macro Precision: {global_segmentation_metrics['macro']['precision']:.4f}")
        print(f"Macro Recall: {global_segmentation_metrics['macro']['recall']:.4f}")
        print(f"Macro F2 Score: {global_segmentation_metrics['macro']['f2']:.4f}")
        print(f"Macro Pk Score: {global_segmentation_metrics['macro']['pk']:.4f}")
        print(f"Macro WindowDiff: {global_segmentation_metrics['macro']['windowdiff']:.4f}")
        print("-----------------")
        print(f"Weighted Precision: {global_segmentation_metrics['weighted']['precision']:.4f}")
        print(f"Weighted Recall: {global_segmentation_metrics['weighted']['recall']:.4f}")
        print(f"Weighted F2 Score: {global_segmentation_metrics['weighted']['f2']:.4f}")
        print(f"Weighted Pk Score: {global_segmentation_metrics['weighted']['pk']:.4f}")
        print(f"Weighted WindowDiff: {global_segmentation_metrics['weighted']['windowdiff']:.4f}")


            
        """
            
        micro_precision, micro_recall, micro_f1 = compute_basic_metrics(
            pred_boundaries_per_class_per_doc[class_name],
            true_boundaries_per_class_per_doc[class_name]
        )
        avg_segment_len = compute_average_segment_length(true_boundaries_per_class_per_doc[class_name], doc_lens)
        global_pk, global_windowdiff = compute_window_metrics(
            pred_boundaries_per_class_per_doc[class_name],
            true_boundaries_per_class_per_doc[class_name],
            doc_lens,
            avg_segment_len / 2
        )
        metrics_per_class[class_name] = {
            "Precision": micro_precision,
            "Recall": micro_recall,
            "F1 Score": micro_f1,
            "1 - Pk Score": 1 - global_pk,
            "1 - WindowDiff": 1 - global_windowdiff
        }

        """
        
    else:
        if task == "BIO":
            pred_boundaries_per_doc, true_boundaries_per_doc = \
                compute_boundaries_BIO(all_preds_textual, all_labels_textual, doc_names)
        elif task == "multiclass":
            pred_boundaries_per_doc, true_boundaries_per_doc = \
                compute_boundaries_multiclass(all_preds_textual, all_labels_textual, doc_names)
        
        micro_precision, micro_recall, micro_f1 = compute_basic_metrics(
            pred_boundaries_per_doc, true_boundaries_per_doc
        )
        avg_segment_len = compute_average_segment_length(true_boundaries_per_doc, doc_lens)
        global_pk, global_windowdiff = compute_window_metrics(
            pred_boundaries_per_doc, true_boundaries_per_doc, doc_lens, avg_segment_len / 2
        )
        
        print("\nSegmentation results")
        print(f"Micro Precision: {micro_precision:.4f}")
        print(f"Micro Recall: {micro_recall:.4f}")
        print(f"Micro F1 Score: {micro_f1:.4f}")
        print(f"Global Pk Score (micro avg): {global_pk:.4f}")
        print(f"Global WindowDiff (micro avg): {global_windowdiff:.4f}")

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# These functions are assumed to be defined elsewhere:
# compute_boundaries_BIO, compute_boundaries_multiclass, compute_basic_metrics,
# compute_average_segment_length, compute_window_metrics

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# These helper functions are assumed to be defined elsewhere:
# compute_boundaries_BIO, compute_boundaries_multiclass, compute_basic_metrics,
# compute_average_segment_length, compute_window_metrics

def evaluate_LLM_from_folders(gt_folder, pred_folder, task):
    """
    Evaluate predictions against ground truth for a set of documents stored as JSON files.

    Each JSON file is expected to be a list of objects with keys "sentence" and "label".
    The prediction and ground truth files should share the same filename.
    
    Parameters:
        gt_folder (str): Folder containing ground truth JSON files.
        pred_folder (str): Folder containing prediction JSON files.
        task (str): "BIO" or "multiclass" (determines which segmentation boundary function to use).
    """
    
    all_preds_with_ids_dict = {}
    all_labels_with_ids_dict = {}
    doc_sentences = {}  # For saving per-document results later
    doc_lens = {}       # To compute segmentation metrics
    
    # Process each prediction file in the folder
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.json')]
    for file in pred_files:
        pred_path = os.path.join(pred_folder, file)
        gt_path = os.path.join(gt_folder, file)
        
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        doc_name = file
        doc_sentences[doc_name] = [item["sentence"] for item in gt_data]
        doc_lens[doc_name] = len(gt_data)
        
        for idx, (pred_item, gt_item) in enumerate(zip(pred_data, gt_data)):
            key = (doc_name, idx)
            all_preds_with_ids_dict[key] = pred_item["label"]
            all_labels_with_ids_dict[key] = gt_item["label"]

    all_preds_textual = all_preds_with_ids_dict
    all_labels_textual = all_labels_with_ids_dict

    #####################
    # Classification Metrics
    #####################
    all_preds_no_window = np.array(list(all_preds_textual.values()))
    all_labels_no_window = np.array(list(all_labels_textual.values()))
    
    class_names = np.unique(np.concatenate([all_preds_no_window, all_labels_no_window]))
    
    accuracy = accuracy_score(all_labels_no_window, all_preds_no_window)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels_no_window, all_preds_no_window, average='macro', zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels_no_window, all_preds_no_window, average='weighted', zero_division=0)
    
    print("\nClassification Results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    
    conf_matrix = confusion_matrix(all_labels_no_window, all_preds_no_window, normalize='true', labels=class_names)
    class_report = classification_report(all_labels_no_window, all_preds_no_window, zero_division=0, labels=class_names, target_names=class_names)
    print("\nClassification Report:\n", class_report)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=False, fmt='.1f', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized by Row)')
    plt.show()

    #####################
    # Segmentation Metrics
    #####################
    # Aggregate doc names from keys
    doc_names_set = {key[0] for key in all_preds_textual.keys()}
    
    if task == "BIO":
        pred_boundaries_per_doc, true_boundaries_per_doc = compute_boundaries_BIO(
            all_preds_textual, all_labels_textual, doc_names_set)
    elif task == "multiclass":
        pred_boundaries_per_doc, true_boundaries_per_doc = compute_boundaries_multiclass(
            all_preds_textual, all_labels_textual, doc_names_set)
    else:
        print("Unrecognized task for segmentation metrics.")
        pred_boundaries_per_doc, true_boundaries_per_doc = {}, {}
    
    micro_precision, micro_recall, micro_f1 = compute_basic_metrics(pred_boundaries_per_doc, true_boundaries_per_doc)
    avg_segment_len = compute_average_segment_length(true_boundaries_per_doc, doc_lens)
    global_pk, global_windowdiff = compute_window_metrics(
        pred_boundaries_per_doc, true_boundaries_per_doc, doc_lens, avg_segment_len / 2)
    
    print("\nSegmentation Results")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Global Pk Score (micro avg): {global_pk:.4f}")
    print(f"Global WindowDiff (micro avg): {global_windowdiff:.4f}")

def load_paths(split_dict_path):
    return json.load(open(split_dict_path, "r", errors="ignore"))

def load_documents(paths):
    """Load documents from a list of paths.

    Args:
        paths (list): List of paths to the documents.

    Returns:
        documents (list): List of dictionaries, each representing a document.
    """
    documents = []
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as file:
            documents.append(json.load(file))
    return documents

def load_documents_with_names(paths):
    """Load documents from a list of paths.

    Args:
        paths (list): List of paths to the documents.

    """
    documents_dict = {}
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as file:
            documents_dict[path] = json.load(file)
    return documents_dict

'''class SequenceClassifier(nn.Module):
    def __init__(self, encoder, num_labels, isTrain=True):
        super(SequenceClassifier, self).__init__()
        self.isTrain = isTrain
        self.encoder = encoder
        self.linear = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        # Freeze embeddings layer of the encoder
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

    def forward(self, input_embeddings, attention_mask, labels=None):
        """Forward pass of the model.

        Args:
            input_embeddings (torch.FloatTensor): (batch_size, seq_len, encoder_hidden_size)
            attention_mask (torch.FloatTensor): (batch_size, seq_len)
            labels (torch.LongTensor, optional): (batch_size, seq_len) token-level labels.
        """
        # Pass embeddings through the encoder
        output_embeddings = self.encoder(
            inputs_embeds=input_embeddings, attention_mask=attention_mask
        ).last_hidden_state  # Shape: (batch_size, seq_len, encoder_hidden_size)

        logits = self.linear(output_embeddings)  # Shape: (batch_size, seq_len, num_labels)
        predictions_list = self.crf.decode(logits, attention_mask.bool())
        padded_predictions = [torch.cat([torch.tensor(seq_pred), torch.full((512 - len(seq_pred),), -100)]) for seq_pred in predictions_list]
        predictions = torch.stack(padded_predictions).to(logits.device)
        
        if self.isTrain:
            tags = torch.where(labels == -100, 0, labels) # change padding label for compatibility with CRF
            loss = -self.crf(logits, tags, attention_mask.bool(), reduction="mean")
            return loss, predictions
        else:
            return predictions

    def freeze_encoder(self):
        """Freeze all encoder parameters except the final pooler layer."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        if hasattr(self.encoder, "pooler"):
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True'''

'''class SequenceClassifier(nn.Module):
    def __init__(self, encoder, num_labels, isTrain=True, class_weights=None):
        super(SequenceClassifier, self).__init__()
        self.isTrain = isTrain
        self.encoder = encoder
        #self.linear = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            #self.loss_fn = CustomLoss(weights=class_weights, alpha=0)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            #self.loss_fn = CustomLoss(alpha=0)

        # Freeze embeddings layer of the encoder
        for param in self.encoder.roberta.embeddings.parameters():
            param.requires_grad = False

    def forward(self, input_embeddings, attention_mask, labels=None):
        """Forward pass of the model.

        Args:
            input_embeddings (torch.FloatTensor): (batch_size, seq_len, encoder_hidden_size)
            attention_mask (torch.FloatTensor): (batch_size, seq_len)
            labels (torch.LongTensor, optional): (batch_size, seq_len) token-level labels.
        """
        # Pass embeddings through the encoder
        output = self.encoder(
            inputs_embeds=input_embeddings, attention_mask=attention_mask
        )  # Shape: (batch_size, seq_len, encoder_hidden_size)

        #logits = self.linear(output_embeddings)  # Shape: (batch_size, seq_len, num_labels)
        logits = output.logits
        softmax_output = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, num_labels)
        predictions = torch.argmax(softmax_output, dim=-1) # Shape: (batch_size, seq_len)
        
        if self.isTrain:
            loss = None
            if labels is not None:
                mask = attention_mask.view(-1) == 1  # Mask for valid tokens
                active_logits = logits.view(-1, logits.size(-1))[mask]
                active_labels = labels.view(-1)[mask]
                #active_embeddings = output_embeddings.view(-1, output_embeddings.size(-1))[mask]
                #loss = self.loss_fn(active_embeddings, active_logits, active_labels)
                loss = self.loss_fn(active_logits, active_labels)
            return loss, predictions
        else:
            return predictions'''

class BiLSTMClassifierMTL(nn.Module):
    def __init__(self, num_labels, isTrain=True, class_weights=None, alpha=1.0, temperature=0.05, binary_loss_coef=0.2):
        super(BiLSTMClassifierMTL, self).__init__()
        self.isTrain = isTrain
        self.binary_loss_coef = binary_loss_coef
        
        self.lstm1 = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, 
                             bidirectional=True, batch_first=True)
        self.ln1 = nn.LayerNorm(768 * 2)
        self.lstm2 = nn.LSTM(input_size=768 * 2, hidden_size=768, num_layers=1, 
                             bidirectional=True, batch_first=True)
        self.ln2 = nn.LayerNorm(768 * 2)
        self.lstm3 = nn.LSTM(input_size=768 * 2, hidden_size=768, num_layers=1, 
                             bidirectional=True, batch_first=True)
        
        # Token-level classification head
        self.linear = nn.Linear(768 * 2, num_labels)
        
        # binary label switch classification head
        # takes the concatenated outputs of the current token and the previous token
        self.binary_head = nn.Linear(768 * 4, 1)
        
        if isTrain:
            self.classification_loss_fn = CustomLoss(alpha=alpha, temperature=temperature, weights=class_weights)
            self.label_switch_loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, input_embeddings, attention_mask, labels=None, binary_labels=None):
        """
        Args:
            input_embeddings (torch.FloatTensor): (batch_size, seq_len, encoder_hidden_size)
            attention_mask (torch.FloatTensor): (batch_size, seq_len)
            labels (torch.LongTensor, optional): (batch_size, seq_len) token-level labels.
            binary_labels (torch.FloatTensor, optional): (batch_size, seq_len - 1) binary labels indicating 
                whether the current and previous outputs are the same (1) or not (0).
        """
        # Compute lengths from attention mask and ensure they're integers
        lengths = attention_mask.sum(dim=-1).long().cpu()
        max_len = input_embeddings.size(1)
        
        # LSTM Layer 1 with packing
        packed_input = pack_padded_sequence(input_embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_output1, _ = self.lstm1(packed_input)
        output1, _ = pad_packed_sequence(packed_output1, batch_first=True, total_length=max_len)
        output1 = self.ln1(output1)
        
        # LSTM Layer 2 with packing
        packed_input2 = pack_padded_sequence(output1, lengths, batch_first=True, enforce_sorted=False)
        packed_output2, _ = self.lstm2(packed_input2)
        output2, _ = pad_packed_sequence(packed_output2, batch_first=True, total_length=max_len)
        output2 = self.ln2(output2)
        
        # LSTM Layer 3 with packing
        packed_input3 = pack_padded_sequence(output2, lengths, batch_first=True, enforce_sorted=False)
        packed_output3, _ = self.lstm3(packed_input3)
        output3, _ = pad_packed_sequence(packed_output3, batch_first=True, total_length=max_len)
        
        # --- Token-level classification head ---
        logits = self.linear(output3)  # (batch, seq_len, num_labels)
        softmax_output = F.softmax(logits, dim=-1)
        predictions = torch.argmax(softmax_output, dim=-1)
        predictions[attention_mask == 0] = -100  # mark padding tokens
        
        # --- Binary classification head ---
        # concatenate outputs of token t (from 1:seq_len) with token t-1 (from 0:seq_len-1)
        binary_input = torch.cat([output3[:, 1:, :], output3[:, :-1, :]], dim=-1)  # (batch, seq_len-1, 768*4)
        binary_logits = self.binary_head(binary_input)  # (batch, seq_len-1, 1)
        binary_logits = binary_logits.squeeze(-1)  # (batch, seq_len-1)
        
        token_loss = 0.0
        binary_loss = 0.0
        
        if self.isTrain:
            # Token-level loss
            if labels is not None:
                mask = attention_mask.view(-1) == 1  # valid tokens mask
                active_embeddings = output3.view(-1, output3.size(-1))[mask]
                active_logits = logits.view(-1, logits.size(-1))[mask]
                active_labels = labels.view(-1)[mask]
                classif_loss = self.classification_loss_fn(active_embeddings, active_logits, active_labels)
            
            # Binary loss
            if labels is not None:
                binary_labels = (labels[:, 1:] == labels[:, :-1]).float()
                # bin mask both consecutive tokens must be non-padding
                binary_mask = (attention_mask[:, 1:] * attention_mask[:, :-1]) == 1
                binary_logits_flat = binary_logits[binary_mask]
                binary_labels_flat = binary_labels[binary_mask]
                binary_loss = self.label_switch_loss_fn(binary_logits_flat, binary_labels_flat)
                
                # threshold the binary predictions
                binary_preds = torch.sigmoid(binary_logits)
                binary_preds = (binary_preds > 0.5).float()
            
            # Combine the losses
            final_loss = classif_loss + self.binary_loss_coef * binary_loss
            return final_loss, predictions, binary_labels, binary_preds
        else:
            return predictions, binary_logits

class BiLSTMMultilabel(nn.Module):
    def __init__(self, num_labels, isTrain=True, class_weights=None):
        super(BiLSTMMultilabel, self).__init__()
        self.isTrain = isTrain
        
        self.lstm1 = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, 
                             bidirectional=True, batch_first=True)
        self.ln1 = nn.LayerNorm(768 * 2)
        self.lstm2 = nn.LSTM(input_size=768 * 2, hidden_size=768, num_layers=1, 
                             bidirectional=True, batch_first=True)
        self.ln2 = nn.LayerNorm(768 * 2)
        self.lstm3 = nn.LSTM(input_size=768 * 2, hidden_size=768, num_layers=1, 
                             bidirectional=True, batch_first=True)
        self.linear = nn.Linear(768 * 2, num_labels)
        
        if isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def forward(self, input_embeddings, attention_mask, labels=None):
        """
        Args:
            input_embeddings (torch.FloatTensor): Tensor of shape (batch_size, seq_len, 768)
            attention_mask (torch.LongTensor): Tensor of shape (batch_size, seq_len) with 1 for valid tokens and 0 for padding.
            labels (torch.FloatTensor, optional): Tensor of shape (batch_size, seq_len, num_labels)
                containing binary labels for multilabel classification.
        """
        # Compute lengths from attention mask
        lengths = attention_mask.sum(dim=-1).long().cpu()
        max_len = input_embeddings.size(1)
        
        # LSTM Layer 1 with packing
        packed_input = pack_padded_sequence(input_embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_output1, _ = self.lstm1(packed_input)
        output1, _ = pad_packed_sequence(packed_output1, batch_first=True, total_length=max_len)
        output1 = self.ln1(output1)
        
        # LSTM Layer 2 with packing
        packed_input2 = pack_padded_sequence(output1, lengths, batch_first=True, enforce_sorted=False)
        packed_output2, _ = self.lstm2(packed_input2)
        output2, _ = pad_packed_sequence(packed_output2, batch_first=True, total_length=max_len)
        output2 = self.ln2(output2)
        
        # LSTM Layer 3 with packing
        packed_input3 = pack_padded_sequence(output2, lengths, batch_first=True, enforce_sorted=False)
        packed_output3, _ = self.lstm3(packed_input3)
        output3, _ = pad_packed_sequence(packed_output3, batch_first=True, total_length=max_len)
        
        logits = self.linear(output3)  # (batch_size, seq_len, num_labels)
        
        # Sigmoid and threshold
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).long()
        
        # Set padding tokens' preds to -100
        predictions[attention_mask == 0] = -100
        
        if self.isTrain and labels is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1, labels.size(-1))
            mask = attention_mask.view(-1) == 1  # valid token positions
            active_logits = logits_flat[mask]
            active_labels = labels_flat[mask]
            
            loss = self.loss_fn(active_logits, active_labels)
            return loss, predictions
        else:
            return predictions
        
class BiLSTMClassifier(nn.Module):
    def __init__(self, num_labels, isTrain=True, class_weights=None, alpha=1, temperature=0.05):
        super(BiLSTMClassifier, self).__init__()
        self.isTrain = isTrain
        
        self.lstm1 = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, bidirectional=True, batch_first=True)
        self.ln1 = nn.LayerNorm(768 * 2)
        self.lstm2 = nn.LSTM(input_size=768 * 2, hidden_size=768, num_layers=1, bidirectional=True, batch_first=True)
        self.ln2 = nn.LayerNorm(768 * 2)
        self.lstm3 = nn.LSTM(input_size=768 * 2, hidden_size=768, num_layers=1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(768 * 2, num_labels)
        
        if isTrain:
            self.loss_fn = CustomLoss(alpha=alpha, temperature=temperature, weights=class_weights)

    def forward(self, input_embeddings, attention_mask, labels=None):
        """
        Args:
            input_embeddings (torch.FloatTensor): (batch_size, seq_len, encoder_hidden_size)
            attention_mask (torch.FloatTensor): (batch_size, seq_len)
            labels (torch.LongTensor, optional): (batch_size, seq_len) token-level labels.
        """
        # Compute actual lengths using the attention mask and convert to integer type
        lengths = attention_mask.sum(dim=-1).long().cpu()
        max_len = input_embeddings.size(1)
        
        # LSTM Layer 1 with packing
        packed_input = pack_padded_sequence(input_embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_output1, _ = self.lstm1(packed_input)
        output1, _ = pad_packed_sequence(packed_output1, batch_first=True, total_length=max_len)
        
        # Apply LayerNorm directly on the output
        output1 = self.ln1(output1)
        
        # LSTM Layer 2
        packed_input2 = pack_padded_sequence(output1, lengths, batch_first=True, enforce_sorted=False)
        packed_output2, _ = self.lstm2(packed_input2)
        output2, _ = pad_packed_sequence(packed_output2, batch_first=True, total_length=max_len)
        
        # Apply second LayerNorm
        output2 = self.ln2(output2)
        
        # LSTM Layer 3
        packed_input3 = pack_padded_sequence(output2, lengths, batch_first=True, enforce_sorted=False)
        packed_output3, _ = self.lstm3(packed_input3)
        output3, _ = pad_packed_sequence(packed_output3, batch_first=True, total_length=max_len)
        
        # Final classification layer
        logits = self.linear(output3)
        softmax_output = F.softmax(logits, dim=-1)
        predictions = torch.argmax(softmax_output, dim=-1)
        predictions[attention_mask == 0] = -100  # mark padding tokens
        
        if self.isTrain:
            loss = None
            if labels is not None:
                mask = attention_mask.view(-1) == 1  # mask valid tokens
                active_embeddings = output3.view(-1, output3.size(-1))[mask]
                active_logits = logits.view(-1, logits.size(-1))[mask]
                active_labels = labels.view(-1)[mask]
                loss = self.loss_fn(active_embeddings, active_logits, active_labels)
            return loss, predictions
        else:
            return predictions

class BiLSTMClassifierOld(nn.Module):
    def __init__(self, num_labels, isTrain=True, class_weights=None, alpha=1, temperature=0.05):
        super(BiLSTMClassifier, self).__init__()
        self.isTrain = isTrain
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=3,
                              batch_first=True, bidirectional=True)
        self.linear = nn.Linear(768 * 2, num_labels)
        
        if isTrain:
            self.loss_fn = CustomLoss(alpha=alpha, temperature=temperature, weights=class_weights)

    def forward(self, input_embeddings, attention_mask, labels=None):
        """Forward pass of the model.

        Args:
            input_embeddings (torch.FloatTensor): (batch_size, seq_len, encoder_hidden_size)
            attention_mask (torch.FloatTensor): (batch_size, seq_len)
            labels (torch.LongTensor, optional): (batch_size, seq_len) token-level labels.
        """
        lengths = attention_mask.sum(dim=-1).cpu()
        packed_embeddings_input = pack_padded_sequence(input_embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_lstm_output, _ = self.lstm(packed_embeddings_input)
        lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True, total_length=512)
        logits = self.linear(lstm_output)
        softmax_output = F.softmax(logits, dim=-1)
        predictions = torch.argmax(softmax_output, dim=-1)
        predictions[attention_mask == 0] = -100 # set padding tokens to -100
        
        if self.isTrain:
            loss = None
            if labels is not None:
                mask = attention_mask.view(-1) == 1  # mask for valid tokens
                active_embeddings = lstm_output.view(-1, lstm_output.size(-1))[mask]
                active_logits = logits.view(-1, logits.size(-1))[mask]
                active_labels = labels.view(-1)[mask]
                loss = self.loss_fn(active_embeddings, active_logits, active_labels)
                
            return loss, predictions
        else:
            return predictions

class BiLSTMCRFClassifier(nn.Module):
    def __init__(self, num_labels, constraints=None, isTrain=True, class_weights=None):
        super(BiLSTMCRFClassifier, self).__init__()
        self.isTrain = isTrain
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=3, 
                              batch_first=True, bidirectional=True, dropout=0.1)
        self.linear = nn.Linear(768 * 2, num_labels)
        self.crf = ConditionalRandomFieldWeightLannoy(num_labels, constraints=constraints, label_weights=class_weights)

    def forward(self, input_embeddings, attention_mask, labels=None):
        """Forward pass of the model.

        Args:
            input_embeddings (torch.FloatTensor): (batch_size, seq_len, encoder_hidden_size)
            attention_mask (torch.FloatTensor): (batch_size, seq_len)
            labels (torch.LongTensor, optional): (batch_size, seq_len) token-level labels.
        """
        # lstm forward
        lengths = attention_mask.sum(dim=-1).cpu()
        packed_embeddings_input = pack_padded_sequence(input_embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_lstm_output, _ = self.lstm(packed_embeddings_input)
        lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True, total_length=512)
        logits = self.linear(lstm_output)
        
        # crf predictions
        best_paths = self.crf.viterbi_tags(logits, attention_mask.bool())
        predictions_list = [path for path, score in best_paths]
        padded_predictions = [torch.cat([torch.tensor(seq_pred), torch.full((512 - len(seq_pred),), -100)]) for seq_pred in predictions_list]
        predictions = torch.stack(padded_predictions).to(logits.device)
        
        # crf forward with loss computation
        if self.isTrain:
            loss = None
            if labels is not None:
                tags = torch.where(labels == -100, 0, labels) # change padding label for compatibility with pytorch-CRF
                loss = - self.crf(logits, tags, attention_mask.bool())
            return loss, predictions
        else:
            return predictions

class SentenceEncoder(nn.Module):
    def __init__(self, encoder):
        super(SentenceEncoder, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():  # freeze training
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """Forward pass of the model

        Args:
            input_ids (torch.LongTensor): (batch_size, seq_len)
            attention_mask (torch.LongTensor): (batch_size, seq_len)

        Returns:
            cls_embeddings (torch.FloatTensor): CLS token embeddings for the batch (batch_size, encoder_hidden_size)
        """
        token_embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # Shape: (batch_size, seq_len, encoder_hidden_size)
        cls_embeddings = token_embeddings[:, 0, :]  # get CLS token embeddings
        return cls_embeddings

class SentenceEncoder2(nn.Module):
    def __init__(self, encoder):
        super(SentenceEncoder2, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():  # freeze training
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model

        Args:
            input_ids (torch.LongTensor): (batch_size, seq_len)
            attention_mask (torch.LongTensor): (batch_size, seq_len)

        Returns:
            mean_embeddings (torch.FloatTensor): Mean-pooled token embeddings for the batch (batch_size, encoder_hidden_size)
        """
        token_embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (batch_size, seq_len, encoder_hidden_size)
        
        # Expand attention_mask to match token_embeddings shape and convert to float
        mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        
        # Multiply token embeddings by mask and sum over sequence length
        sum_embeddings = (token_embeddings * mask).sum(dim=1)  # (batch_size, encoder_hidden_size)
        
        # Compute the number of valid tokens per sample and avoid division by zero.
        lengths = mask.sum(dim=1)  # (batch_size, 1)
        lengths = lengths.clamp(min=1e-9)
        
        mean_embeddings = sum_embeddings / lengths  # (batch_size, encoder_hidden_size)
        return mean_embeddings
    
class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        documents,
        tokenizer,
        encoder,
        device,
        labels_dict,
        task,
        encoder_batch_size = 256,
        return_labels = False,
        max_sentences = 512,
        max_sentence_len = 512,
        overlap_percent = 50,
    ):
        """
        Args:
            documents (dict): Dictionary of documents, where each document is a list of dictionaries containing the sentence and labels.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for tokenization.
            encoder (transformers.PreTrainedModel): Encoder to be used for extracting embeddings from the sentences.
            device (torch.device): Device to be used for creating the sequences.
            labels_dict (dict): Dictionary mapping label strings to integers.
            task (str): "multilabel" or "multiclass".
            return_labels (bool): Whether to return labels or not, if inference is being done, this is set to False.
            max_sentences (int): Maximum number of sentences in a sequence.
            max_sentence_len (int): Maximum number of tokens in a sentence.
        """
        self.return_labels = return_labels
        self.documents = documents
        self.labels_dict = labels_dict
        self.task = task
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.encoder_batch_size = encoder_batch_size
        self.device = device
        self.max_sentences = max_sentences
        self.max_sentence_len = max_sentence_len
        self.overlap_percent = overlap_percent
        
        self.encoder.to(self.device)
        self.encoder.eval()
        with torch.no_grad():
            self.sequence_embeddings, self.ids_by_sentence, self.attention_masks, self.labels = self._generate_sequences_and_labels()

    def _generate_sequences_and_labels(self):
        """
        Generate sequences from the documents.
        
        Returns:
            sequences (torch.Tensor): Tensor of sequences of embeddings. 
                                      Shape: (num_sequences, max_sentences, encoder_hidden_size)
            ids_by_sentence (list): Sentence ids for each sequence, where each id is composed of document name and line number.
            labels (torch.Tensor): Tensor of labels. For multiclass: (num_sequences, max_sentences); 
                                   for multilabel: (num_sequences, max_sentences, num_labels).
            attention_masks (torch.Tensor): Tensor of attention masks. Shape: (num_sequences, max_sentences)
        """
        sequences = []
        ids_by_sentence = []
        labels = []
        attention_masks = []

        for doc_name, doc in tqdm(self.documents.items(), unit="document"):
            stride = int(self.max_sentences * (1 - self.overlap_percent / 100))

            for start in range(0, len(doc), stride):
                end = start + self.max_sentences
                raw_sequence = doc[start:end]
                batched_raw_sequence = [
                    raw_sequence[i : i + self.encoder_batch_size]
                    for i in range(0, len(raw_sequence), self.encoder_batch_size)
                ]
                
                sequence = []
                # Build sentence IDs for this sequence (using the actual number of sentences in this sequence)
                ids_insequence = [[doc_name, line_index] for line_index in range(start, min(end, len(doc)))]
                labels_insequence = []
                
                for batch in batched_raw_sequence:
                    # Get sentences from the batch
                    sentences_raw = [item["sentence"] for item in batch]
                    
                    # Tokenize batch
                    tokenized_batch = self.tokenizer(
                        sentences_raw,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_sentence_len,
                    )
                    input_ids_batch = tokenized_batch["input_ids"].to(self.device)
                    attention_mask_batch = tokenized_batch["attention_mask"].to(self.device)
                    
                    # Get embeddings from the encoder (assumed to return a tensor of shape [batch_size, encoder_hidden_size])
                    embeddings_batch = self.encoder(input_ids_batch, attention_mask_batch).detach().cpu()
                    
                    # Append embeddings; iterating over the batch ensures each element is a tensor.
                    for emb in embeddings_batch:
                        sequence.append(emb)
                    
                    if self.return_labels:
                        if self.task == "multiclass":
                            # For multiclass, process each sentence's label as a scalar tensor.
                            for item in batch:
                                # Here, item["label"] should be mapped via labels_dict to an int.
                                labels_insequence.append(torch.tensor(self.labels_dict[item["label"]]))
                        elif self.task == "multilabel":
                            # For multilabel, process each sentence's labels into a multi-hot tensor
                            for item in batch:
                                label_list = item["labels"]
                                one_hot = torch.zeros(len(self.labels_dict))
                                for label in label_list:
                                    if label == 'Other':
                                        continue
                                    one_hot[self.labels_dict[label]] = 1
                                labels_insequence.append(one_hot)
                
                # Build the attention mask for this sequence (for the number of sentences actually processed)
                current_seq_len = len(sequence)
                att_mask = torch.cat(
                    [torch.ones(current_seq_len), torch.zeros(self.max_sentences - current_seq_len)]
                )
                attention_masks.append(att_mask)
                
                # Pad the sequence (embeddings) if necessary to have exactly max_sentences entries
                if current_seq_len < self.max_sentences:
                    pad_tensor = torch.zeros_like(sequence[0])
                    for _ in range(self.max_sentences - current_seq_len):
                        sequence.append(pad_tensor)
                sequence = torch.stack(sequence)  # Shape: (max_sentences, encoder_hidden_size)
                sequences.append(sequence)
                ids_by_sentence.append(ids_insequence)
                
                if self.return_labels:
                    # Pad labels to have self.max_sentences entries
                    if self.task == "multiclass":
                        while len(labels_insequence) < self.max_sentences:
                            labels_insequence.append(torch.tensor(-100))
                    elif self.task == "multilabel":
                        while len(labels_insequence) < self.max_sentences:
                            pad_label = torch.full((len(self.labels_dict),), -100)
                            labels_insequence.append(pad_label)
                    labels.append(torch.stack(labels_insequence))
                
                if end >= len(doc):
                    break
            
        if self.return_labels:
            return torch.stack(sequences), ids_by_sentence, torch.stack(attention_masks), torch.stack(labels)
        else:
            return torch.stack(sequences), ids_by_sentence, torch.stack(attention_masks), None

    def save(self, path):
        """
        Save the dataset state to disk using pickle.
        """
        state = {
            "documents": self.documents,
            "encoder_batch_size": self.encoder_batch_size,
            "max_sentences": self.max_sentences,
            "max_sentence_len": self.max_sentence_len,
            "overlap_percent": self.overlap_percent,
            "sequence_embeddings": self.sequence_embeddings,
            "ids_by_sentence": self.ids_by_sentence,
            "attention_masks": self.attention_masks,
            "labels": self.labels
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Dataset saved to {path}")
    
    @classmethod
    def load(cls, path, tokenizer=None, encoder=None, device=None):
        """
        Load the dataset state from disk and re-create a TrainingDataset instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        # Create a new instance without triggering _generate_sequences()
        obj = cls.__new__(cls)
        obj.documents = state["documents"]
        obj.tokenizer = tokenizer
        obj.encoder = encoder
        obj.device = device
        obj.encoder_batch_size = state["encoder_batch_size"]
        obj.max_sentences = state["max_sentences"]
        obj.max_sentence_len = state["max_sentence_len"]
        obj.overlap_percent = state["overlap_percent"]
        obj.sequence_embeddings = state["sequence_embeddings"]
        obj.ids_by_sentence = state["ids_by_sentence"]
        obj.attention_masks = state["attention_masks"]
        obj.labels = state["labels"]
        
        print(f"Dataset loaded from {path}")
        print(f"Number of sequences loaded: {len(obj.sequence_embeddings)}\n")
        return obj
    
    def __len__(self):
        return len(self.sequence_embeddings)
        
    def __getitem__(self, idx):
        return {
            "input_embeddings": self.sequence_embeddings[idx],  # Shape: (max_sentence_len, encoder_hidden_size)
            "sentence_ids": self.ids_by_sentence[idx],  # Shape: (max_sentence_len, 2)
            "attention_mask": self.attention_masks[idx],  # Shape: (max_sentence_len)
            "labels": self.labels[idx],  # Shape: (max_sentence_len) or (max_sentence_len, num_labels)
        }

def custom_collator(batch):
    """
    Custom collation function to keep 'sentence_ids' as a list and
    batch other items as tensors.
    """
    input_embeddings = torch.stack([item["input_embeddings"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    sentence_ids = [item["sentence_ids"] for item in batch]

    return {
        "input_embeddings": input_embeddings,
        "sentence_ids": sentence_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
        
def load_crf_constraints(constraints_path):
    loaded_constraints = json.load(open(constraints_path, "r", errors="ignore"))
    loaded_constraints = [tuple(item) for item in loaded_constraints]
    return loaded_constraints
    
def compute_label_distribution(train_documents, labels_dict, task):
    labels_count = Counter(labels_dict.keys())
    for doc_name, doc in train_documents.items():
        for sentence in doc:
            if task == "multiclass":
                labels_count.update([sentence["label"]])
            elif task == "multilabel":
                labels_count.update(sentence["labels"])
                
    #normalizing the counts
    total = sum(labels_count.values())
    for label in labels_count.keys():
        labels_count[label] /= total
    
    return torch.tensor(list(labels_count.values()))

def compute_class_weights(train_documents, labels_dict, task):
    if task == "multiclass":
        labels_count = Counter(labels_dict.keys())
        # empty counts
        for key in labels_count.keys():
            labels_count[key] = 0
        
        for doc_name, doc in train_documents.items():
            for sentence in doc:
                labels_count.update([sentence["label"]])
        
        # normalize counts
        total = sum(labels_count.values())
        for label in labels_count.keys():
            labels_count[label] /= total
        
        class_weights = torch.zeros(len(labels_dict))
        
        # compute class weights, setting 0 for classes with no samples
        for i, dist_value in enumerate(labels_count.values()):
            if dist_value > 0:
                class_weights[i] = 1 / dist_value
            else:
                class_weights[i] = 0
            
        class_weights = class_weights / class_weights.sum()
    elif task == "multilabel": # weight = Number of samples where class i is not present / Number of samples where class i is present
        present_counts = Counter(labels_dict.keys())
        notpresent_counts = Counter(labels_dict.keys())
        
        # empty counts
        for key in notpresent_counts.keys():
            present_counts[key] = 0
            notpresent_counts[key] = 0
            
        for doc_name, doc in train_documents.items():
            for sentence in doc:
                present_labels = sentence["labels"]
                notpresent_labels = list(set(labels_dict.keys()) - set(present_labels))
                present_counts.update(present_labels)
                notpresent_counts.update(notpresent_labels)
        
        # compute class weights, setting 0 for classes with no samples
        class_weights = torch.zeros(len(labels_dict))
        for i, label in enumerate(labels_dict.keys()):
            present = present_counts[label]
            notpresent = notpresent_counts[label]
            if present > 0:
                class_weights[i] = notpresent / present
            else:
                class_weights[i] = 0
        
    return class_weights    

class CustomLoss(nn.Module):
    def __init__(self, alpha=1, temperature=0.05, weights=None):
        """
        Args:
            alpha (float): Weight for the supervised contrastive loss.
            temperature (float): Temperature scaling for contrastive loss.
            weights (torch.Tensor): Class weights for the cross-entropy loss.
        """
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.contrastive_loss_fn = SupConLoss(temperature=temperature)
        self.focal_loss_fn = FocalLoss(class_weights=weights)
    
    def forward(self, embeddings, logits, labels):
        """
        Computes the combined loss.

        Args:
            embeddings (torch.Tensor): Embedding vectors of shape [batch_size, embedding_dim].
            logits (torch.Tensor): Logits of shape [batch_size, num_classes].
            labels (torch.Tensor): Ground-truth labels of shape [batch_size].

        Returns:
            torch.Tensor: Combined loss (scalar).
        """
        # Normalize embeddings for supervised contrastive loss
        embeddings = F.normalize(embeddings, dim=1)

        # Compute supervised contrastive loss
        contrastive_loss = self.contrastive_loss_fn(embeddings, labels)

        # Compute focal loss
        focal_loss = self.focal_loss_fn(logits, labels)

        # Combine the two losses
        total_loss = self.alpha * contrastive_loss + focal_loss
        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='sum', class_weights=None):
        super(FocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=-1)
        probs = probs.clamp(min=1e-12, max=1.0)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        probs_true = (probs * targets_one_hot).sum(dim=1)

        focal_weight = (1 - probs_true) ** self.gamma
        
        log_probs = torch.log(probs_true)

        if self.class_weights == None:
            class_weights = 1
        else:
            class_weights = self.class_weights[targets]
        
        loss = -class_weights * focal_weight * log_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss