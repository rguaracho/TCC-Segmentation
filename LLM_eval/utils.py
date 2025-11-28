# Generate train, validation and test split
# sequences with a given window size
import json
import torch
import numpy as np
import torch.nn.functional as F
#import evaluate
from torch.utils.data import Dataset
from torch import nn
from tqdm.notebook import tqdm
from pytorch_metric_learning.losses import SupConLoss
#from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from allennlp.modules.conditional_random_field import ConditionalRandomFieldWeightLannoy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#accuracy_metric = evaluate.load("accuracy")
#precision_metric = evaluate.load("precision")
#recall_metric = evaluate.load("recall")
#f1_metric = evaluate.load("f1")

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

def compute_basic_metrics(pred_boundaries_per_doc, true_boundaries_per_doc):
    """
    Compute micro-averaged precision, recall, and F1 score over multiple documents.
    The labels are the boundaries themselves, and the boundaries are the indices of lines.
    """
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
    
    micro_precision = total_true_positives / total_predicted if total_predicted > 0 else 0.0
    micro_recall = total_true_positives / total_actual if total_actual > 0 else 0.0
    
    if micro_precision + micro_recall == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    
    return micro_precision, micro_recall, micro_f1

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
            
            preds = model(input_embeddings, attention_mask=attention_mask)
            mask = attention_mask.to('cpu').numpy().astype(bool)
            
            preds_with_ids = []
            labels_with_ids = []
            for i in range(preds.shape[0]):
                for j in range(preds.shape[1]):
                    if mask[i][j]:
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
    
    # convert label codes to textual classes
    code2label = {v: k for k, v in labels_dict.items()}
    all_preds_textual = {k: code2label[v] for k, v in all_preds_with_ids_dict.items()}
    all_labels_textual = {k: code2label[v] for k, v in all_labels_with_ids_dict.items()}

    for doc_name in doc_names:
        doc_results = []
        with open(doc_name, "r", encoding="utf-8-sig", errors="ignore") as file:
                doc_content = json.load(file)
        for key, predicted_label in all_preds_textual.items():
            if key[0] == doc_name:
                sentence = doc_content[key[1]]["sentence"]
                doc_results.append({"sentence": sentence, "predicted_label": predicted_label})
        json.dump(doc_results, open(save_path + f"/{doc_name.split('/')[-1]}", "w"), indent=4, ensure_ascii=False)

    ########### Classification metrics (no window superposition) ##########
    print("\nClassification results")
    all_preds_no_window = np.array(list(all_preds_with_ids_dict.values()))
    all_labels_no_window = np.array(list(all_labels_with_ids_dict.values()))

    unique_labels = np.unique(np.concatenate([all_labels_no_window, all_preds_no_window]))
    class_names = [k for k, v in labels_dict.items() if v in unique_labels]

    accuracy = accuracy_score(all_labels_no_window, all_preds_no_window)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels_no_window, all_preds_no_window, average='macro', zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_labels_no_window, all_preds_no_window, average='weighted', zero_division=0)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

    conf_matrix = confusion_matrix(all_labels_no_window, all_preds_no_window, normalize='true', labels=unique_labels)

    class_report = classification_report(all_labels_no_window, all_preds_no_window, zero_division=0, labels = unique_labels, target_names=class_names)
    print(class_report)

    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=False, fmt='.1f', cmap='viridis', xticklabels=class_names, yticklabels=class_names, linewidths=.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix, normalized by row')
    plt.show()

    ######### Segmentation metrics ##########
    if task == "BIO":
        pred_boundaries_per_doc, true_boundaries_per_doc = compute_boundaries_BIO(all_preds_textual, all_labels_textual, doc_names)
    if task == "multiclass":
        pred_boundaries_per_doc, true_boundaries_per_doc = compute_boundaries_multiclass(all_preds_textual, all_labels_textual, doc_names)
    
    # compute global micro metrics which are strict to boundary position
    micro_precision, micro_recall, micro_f1 = compute_basic_metrics(pred_boundaries_per_doc, true_boundaries_per_doc)
    
    # compute global pk-score and windowdiff metrics which are less strict to boundary position
    doc_lens = {}
    for doc in doc_names:
        with open(doc, "r", encoding="utf-8-sig", errors="ignore") as file:
            doc_lens[doc] = len(json.load(file))
    avg_segment_len = compute_average_segment_length(true_boundaries_per_doc, doc_lens)
    global_pk, global_windowdiff = compute_window_metrics(pred_boundaries_per_doc, true_boundaries_per_doc, doc_lens, avg_segment_len/2)
    
    print("\nSegmentation results\n")
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

def evaluate_LLM_from_folders(gt_folder, pred_folder, save_folder, task, labels_dict=None):
    """
    Evaluate predictions against ground truth for a set of documents stored as JSON files.

    Each JSON file is expected to be a list of objects with keys "sentence" and "label".
    The prediction and ground truth files should share the same filename.
    
    Parameters:
        gt_folder (str): Folder containing ground truth JSON files.
        pred_folder (str): Folder containing prediction JSON files.
        save_folder (str): Folder where per-document prediction results will be saved.
        task (str): "BIO" or "multiclass" (determines which segmentation boundary function to use).
        labels_dict (dict, optional): Mapping from textual label to numeric code.
                                      If provided, it is used to convert codes to labels.
    """
    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    all_preds_with_ids_dict = {}
    all_labels_with_ids_dict = {}
    doc_sentences = {}  # For saving per-document results later
    doc_lens = {}       # To compute segmentation metrics
    
    # Process each prediction file in the folder
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.json')]
    for file in pred_files:
        pred_path = os.path.join(pred_folder, file)
        gt_path = os.path.join(gt_folder, file)  # Assumes the same filename in both folders
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file for {file} not found. Skipping.")
            continue
        
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        doc_name = file  # or you might want to use os.path.splitext(file)[0]
        doc_sentences[doc_name] = [item["sentence"] for item in gt_data]
        doc_lens[doc_name] = len(gt_data)
        
        # Assume that the number of sentences in both files is the same
        for idx, (pred_item, gt_item) in enumerate(zip(pred_data, gt_data)):
            key = (doc_name, idx)
            all_preds_with_ids_dict[key] = pred_item["label"]
            all_labels_with_ids_dict[key] = gt_item["label"]

    # Optionally convert label codes to textual labels if a labels_dict is provided
    if labels_dict is not None:
        code2label = {v: k for k, v in labels_dict.items()}
        all_preds_textual = {k: code2label.get(v, v) for k, v in all_preds_with_ids_dict.items()}
        all_labels_textual = {k: code2label.get(v, v) for k, v in all_labels_with_ids_dict.items()}
    else:
        all_preds_textual = all_preds_with_ids_dict
        all_labels_textual = all_labels_with_ids_dict

    #####################
    # Classification Metrics
    #####################
    all_preds_no_window = np.array(list(all_preds_textual.values()))
    all_labels_no_window = np.array(list(all_labels_textual.values()))
    
    unique_labels = np.unique(np.concatenate([all_preds_no_window, all_labels_no_window]))
    if labels_dict is not None:
        # Use the order defined in labels_dict if possible
        class_names = [k for k, v in labels_dict.items() if v in unique_labels]
    else:
        class_names = sorted(unique_labels)
    
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
    
    conf_matrix = confusion_matrix(all_labels_no_window, all_preds_no_window, normalize='true', labels=unique_labels)
    class_report = classification_report(
        all_labels_no_window, all_preds_no_window, zero_division=0, labels=unique_labels, target_names=class_names)
    print("\nClassification Report:\n", class_report)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=False, fmt='.1f', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized by Row)')
    plt.show()

    # Save per-document predictions with sentences for further inspection
    for doc_name, sentences in doc_sentences.items():
        doc_results = []
        for idx, sentence in enumerate(sentences):
            key = (doc_name, idx)
            predicted_label = all_preds_textual.get(key, 'O')
            doc_results.append({"sentence": sentence, "predicted_label": predicted_label})
        save_file = os.path.join(save_folder, doc_name)
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(doc_results, f, indent=4, ensure_ascii=False)

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
            
            # Combine the losses
            final_loss = classif_loss + self.binary_loss_coef * binary_loss
            return final_loss, predictions
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

class InferenceDataset(Dataset):
    def __init__(
        self,
        documents,
        tokenizer,
        encoder,
        device,
        labels_dict,
        encoder_batch_size = 256,
        return_labels = False,
        max_sentences = 512,
        max_sentence_len = 512,
    ):
        """
        Args:
            documents (dict): Dictionary of documents, where each document is a list of dictionaries containing the sentence and labels.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for tokenization.
            encoder (transformers.PreTrainedModel): Encoder to be used for extracting embeddings from the sentences.
            device (torch.device): Device to be used for creating the sequences.
            labels_dict (dict): Dictionary mapping label strings to integers.
            return_labels (bool): Whether to return labels or not, if inference is being done, this is set to False.
            max_sentences (int): Maximum number of sentences in a sequence.
            max_sentence_len (int): Maximum number of tokens in a sentence.
        """
        self.return_labels = return_labels
        self.documents = documents
        self.labels_dict = labels_dict
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.encoder_batch_size = encoder_batch_size
        self.device = device
        self.max_sentences = max_sentences
        self.max_sentence_len = max_sentence_len
        
        self.encoder.to(self.device)
        self.encoder.eval()
        with torch.no_grad():
            self.sequences, self.ids_by_sentence, self.attention_masks, self.labels = self._generate_sequences_and_labels()

    def _generate_sequences_and_labels(self, overlap_percent=50):
        """
        Generate sequences from the documents.

        Args:
            overlap_percent (int): Percentage of overlap between sequences.
        Returns:
            sequences (torch.Tensor): Tensor of sequences of embeddings. Shape: (num_sequences, max_sentences, encoder_hidden_size)
            ids_by_sentence (list): Sentence ids for each sequence, each id is composed by document name and line number Shape: (num_sequences, max_sentences, 2)
            labels (torch.Tensor): Tensor of labels. Shape: (num_sequences, max_sentences)
            attention_masks (torch.Tensor): Tensor of attention masks. Shape: (num_sequences, max_sentences)
        """
        sequences = []
        ids_by_sentence = []
        labels = []
        attention_masks = []

        for doc_name, doc in tqdm(self.documents.items(), unit="document"):
            stride = int(self.max_sentences * (1 - overlap_percent / 100))

            for start in range(0, len(doc), stride):
                end = start + self.max_sentences
                raw_sequence = doc[start:end]
                batched_raw_sequence = [raw_sequence[i:i + self.encoder_batch_size] for i in range(0, len(raw_sequence), self.encoder_batch_size)]
                
                sequence = []
                ids_insequence = [[doc_name, line_index] for line_index in range(start, end)]
                labels_insequence = []
                
                
                for batch in batched_raw_sequence:
                    sentences_raw = [item["sentence"] for item in batch]
                    
                    tokenized_batch = self.tokenizer(
                        sentences_raw,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_sentence_len,
                    )
                    input_ids_batch = tokenized_batch["input_ids"].to(self.device)
                    attention_mask_batch = tokenized_batch["attention_mask"].to(self.device)
                    
                    # forward pass through the embedding model to get CLS token embeddings: returns size[batch_size, encoder_hidden_size]
                    embeddings_batch = self.encoder(input_ids_batch, attention_mask_batch).detach().cpu()
                    
                    # append the embeddings to the sequence
                    sequence.extend(embeddings_batch)
                    
                    if self.return_labels:
                        # create labels tensor, no one-hot encoding
                        labels_raw = [item["label"] for item in batch]
                        labels_tensor = torch.tensor([self.labels_dict[label] for label in labels_raw])
                            
                        # append labels to the sequence
                        labels_insequence.extend(labels_tensor)

                # append the attention mask to the list
                attention_mask = torch.cat(
                    [
                        torch.ones(len(sequence)),
                        torch.zeros(self.max_sentence_len - len(sequence)),
                    ]
                )
                attention_masks.append(attention_mask)

                # append the sequence to the list (padding if necessary)
                if len(sequence) < self.max_sentence_len:
                    sequence.extend(
                        [
                            torch.zeros_like(sequence[0])
                            for _ in range(self.max_sentences - len(sequence))
                        ]
                    )

                sequence = torch.stack(sequence)  # convert list of tensors to tensor
                sequences.append(sequence)
                ids_by_sentence.append(ids_insequence)

                if self.return_labels:
                    # append the labels to the labels list, padding with -100 if necessary
                    if len(labels_insequence) < self.max_sentence_len:
                        # pad the labels with -100
                        labels_insequence.extend(
                            torch.tensor([-100 for _ in range(self.max_sentence_len - len(labels_insequence))])
                        )

                    labels_insequence = torch.stack(
                        labels_insequence
                    )  # convert list of tensors to tensor
                    labels.append(labels_insequence)

                if end >= len(doc):
                    break

        if self.return_labels:
            return torch.stack(sequences), ids_by_sentence, torch.stack(attention_masks), torch.stack(labels)
        else:
            return torch.stack(sequences), ids_by_sentence, torch.stack(attention_masks), None

    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return {
            "input_embeddings": self.sequences[idx],  # Shape: (max_sentence_len, encoder_hidden_size)
            "sentence_ids": self.ids_by_sentence[idx],  # Shape: (max_sentence_len, 2)
            "attention_mask": self.attention_masks[idx],  # Shape: (max_sentence_len)
            "labels": self.labels[idx],  # Shape: (max_sentence_len)
        }

def inference_collator(batch):
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

class SequencesDataset(Dataset):
    def __init__(
        self,
        documents,
        tokenizer,
        encoder,
        device,
        labels_dict,
        encoder_batch_size = 256,
        return_labels = False,
        max_sentences = 512,
        max_sentence_len = 512,
        overlap_percent = 80,
    ):
        """
        Args:
            documents (list): List of documents, where each document is a list of dictionaries containing the sentence and labels.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for tokenization.
            encoder (transformers.PreTrainedModel): Encoder to be used for extracting embeddings from the sentences.
            device (torch.device): Device to be used for creating the sequences.
            labels_dict (dict): Dictionary mapping label strings to integers.
            return_labels (bool): Whether to return labels or not, if inference is being done, this is set to False.
            max_sentences (int): Maximum number of sentences in a sequence.
            max_sentence_len (int): Maximum number of tokens in a sentence.
        """
        self.return_labels = return_labels
        self.documents = documents
        self.labels_dict = labels_dict
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.encoder_batch_size = encoder_batch_size
        self.device = device
        self.max_sentences = max_sentences
        self.max_sentence_len = max_sentence_len
        self.overlap_percent = overlap_percent
        
        # encode the document line sequences
        self.encoder.to(self.device)
        self.encoder.eval()
        with torch.no_grad():
            self.sequences, self.attention_masks, self.labels = self._generate_sequences_and_labels()
        print(f"Number of sequences encoded: {len(self.sequences)}\n")
        
    def _generate_sequences_and_labels(self):
        """
        Generate sequences from the documents.

        Args:
            overlap_percent (int): Percentage of overlap between sequences.
        Returns:
            sequences (torch.Tensor): Tensor of sequences of embeddings. Shape: (num_sequences, max_sentence_len, encoder_hidden_size)
            labels (torch.Tensor): Tensor of labels. Shape: (num_sequences, max_sentence_len)
            attention_masks (torch.Tensor): Tensor of attention masks. Shape: (num_sequences, max_sentence_len)
        """
        sequences = []
        labels = []
        attention_masks = []

        for doc in tqdm(self.documents, unit="document"):
            stride = int(self.max_sentences * (1 - self.overlap_percent / 100))

            for start in range(0, len(doc), stride):
                end = start + self.max_sentences
                raw_sequence = doc[start:end]
                batched_raw_sequence = [raw_sequence[i:i + self.encoder_batch_size] for i in range(0, len(raw_sequence), self.encoder_batch_size)]
                
                sequence = []
                labels_insequence = []
                
                for batch in batched_raw_sequence:
                    sentences_raw = [item["sentence"] for item in batch]
                    tokenized_batch = self.tokenizer(
                        sentences_raw,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_sentence_len,
                    )
                    input_ids_batch = tokenized_batch["input_ids"].to(self.device)
                    attention_mask_batch = tokenized_batch["attention_mask"].to(self.device)
                    
                    # forward pass through the embedding model to get CLS token embeddings: returns size[batch_size, encoder_hidden_size]
                    embeddings_batch = self.encoder(input_ids_batch, attention_mask_batch).detach().cpu()
                    
                    # append the embeddings to the sequence
                    sequence.extend(embeddings_batch)
                    
                    if self.return_labels:
                        # create labels tensor, no one-hot encoding
                        labels_raw = [item["label"] for item in batch]
                        labels_tensor = torch.tensor([self.labels_dict[label] for label in labels_raw])
                            
                        # append labels to the sequence
                        labels_insequence.extend(labels_tensor)

                # append the attention mask to the list
                attention_mask = torch.cat(
                    [
                        torch.ones(len(sequence)),
                        torch.zeros(self.max_sentence_len - len(sequence)),
                    ]
                )
                attention_masks.append(attention_mask)

                # append the sequence to the list (padding if necessary)
                if len(sequence) < self.max_sentence_len:
                    sequence.extend(
                        [
                            torch.zeros_like(sequence[0])
                            for _ in range(self.max_sentences - len(sequence))
                        ]
                    )

                sequence = torch.stack(sequence)  # convert list of tensors to tensor
                sequences.append(sequence)

                if self.return_labels:
                    # append the labels to the labels list, padding with -100 if necessary
                    if len(labels_insequence) < self.max_sentence_len:
                        # pad the labels with -100
                        labels_insequence.extend(
                            torch.tensor([-100 for _ in range(self.max_sentence_len - len(labels_insequence))])
                        )

                    labels_insequence = torch.stack(
                        labels_insequence
                    )  # convert list of tensors to tensor
                    labels.append(labels_insequence)

                if end >= len(doc):
                    break

        if self.return_labels:
            return (
                torch.stack(sequences),
                torch.stack(attention_masks),
                torch.stack(labels),
            )
        else:
            return torch.stack(sequences), torch.stack(attention_masks)

    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return {
            "input_embeddings": self.sequences[idx],  # Shape: (max_sentence_len, encoder_hidden_size)
            "attention_mask": self.attention_masks[idx],  # Shape: (max_sentence_len)
            "labels": self.labels[idx],  # Shape: (max_sentence_len)
        }

def load_crf_constraints(constraints_path):
    loaded_constraints = json.load(open(constraints_path, "r", errors="ignore"))
    loaded_constraints = [tuple(item) for item in loaded_constraints]
    return loaded_constraints
 
'''def compute_metrics(predictions, labels):    
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    attention_mask = labels != -100
    labels = labels[attention_mask]

    predictions = np.array([pred for pred in predictions])
    predictions = predictions[attention_mask]

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro", zero_division=0)
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro", zero_division=0)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }'''
    
def compute_label_distribution(train_documents, labels_dict):
    labels_count = {label: 0 for label in labels_dict.keys()}
    for doc in train_documents:
        for sentence in doc:
            if sentence["label"] != None:
                labels_count[sentence["label"]] += 1
                
    #normalizing the counts
    total = sum(labels_count.values())
    for label in labels_count.keys():
        labels_count[label] /= total
    
    return torch.tensor(list(labels_count.values()))

def compute_class_weights(train_documents, labels_dict):
    labels_count = {label: 0 for label in labels_dict.keys()}
    for doc in train_documents:
        for sentence in doc:
            if sentence["label"] != None:
                labels_count[sentence["label"]] += 1
                
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