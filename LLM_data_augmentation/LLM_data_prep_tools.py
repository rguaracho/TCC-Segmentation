import os
import json
import re
import pandas as pd

def get_files(path, file_type):
    sort_func = lambda filename: int(''.join(filter(str.isdigit, filename)))
    return sorted([f for f in os.listdir(path) if f.endswith(file_type)], key=sort_func)

def clean_line(text):
    # Remove control characters and unwanted symbols but keep Latin characters with accents
    # This regex keeps letters (including accents), numbers, spaces, and basic punctuation
    cleaned_text = re.sub(r"[^\w\s.,!?\'\"àâçéèêëîïôûùüÿœæÀÂÇÉÈÊËÎÏÔÛÙÜŸŒÆ•():;-]", "", text)
    # Optionally, collapse multiple spaces into one and strip leading/trailing spaces
    return " ".join(cleaned_text.split())

def unify_text_and_labels_multiclass_LLM_hierarchy(texts_path, labels_path, save_path):
    """Unify text and labels into a single JSON file.
    The labelling approach is a multiclass {BIO + hierarchy}.
    It also generates new .txt files with the treatment that was applied to the text.

    Args:
        texts_path (str): Path to the directory containing the text files.
        labels_path (str): Path to the directory containing the label files.
        save_path (str): Path to the directory where the unified json files will be saved.
    """
    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')
    for i_file in range(len(label_files)):
        dict_label_range = {}
        with open(labels_path + '/' + label_files[i_file], 'r') as file:
            file_content = json.load(file)
            for block in file_content:
                label = block['block_type']
                if label in dict_label_range:
                    dict_label_range[label].append((block['start_line'] - 1, block['end_line'] - 1))
                else:
                    dict_label_range[label] = [(block['start_line'] - 1, block['end_line'] - 1)]
        value_hierarchie_single = [{tup: 1} for sublist in dict_label_range.values() for tup in sublist]
        for i in range(len(value_hierarchie_single)):
            current_value = value_hierarchie_single[i]
            current_start, current_end = list(current_value.keys())[0]
            for j in range(len(value_hierarchie_single)):
                if i == j:
                    continue
                potential_value = value_hierarchie_single[j]
                potential_start, potential_end = list(potential_value.keys())[0]
                if (current_start >= potential_start) and (current_end <= potential_end):
                    value_hierarchie_single[i][(current_start, current_end)] += 1
        
        doc = pd.DataFrame(columns=['sentence', 'label'])
        with open(texts_path + '/' + text_files[i_file], 'r', encoding="utf-8-sig", errors='ignore') as file:
            lines = file.readlines()
            lines = [clean_line(line) for line in lines]
            line_id = 0
            for line in lines:
                current_max = 0
                current_interval = (0, 0)
                for i in range(len(value_hierarchie_single)):
                    current_value = value_hierarchie_single[i]
                    current_start, current_end = list(current_value.keys())[0]
                    if (line_id >= current_start) and (line_id <= current_end):
                        if value_hierarchie_single[i][(current_start, current_end)] >= current_max:
                            current_max = value_hierarchie_single[i][(current_start, current_end)]
                            current_interval = (current_start, current_end)
                for label in dict_label_range:
                    if current_interval in dict_label_range[label]:
                        current_label = label
                if current_max == 0 or current_label == 'Other':
                    doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': 'O'})], ignore_index=True)
                else:
                    if line_id == current_interval[0]:
                        doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'B-{current_label}-{current_max}'})], ignore_index=True)
                    else:
                        doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'I-{current_label}-{current_max}'})], ignore_index=True)
                line_id +=1
        doc['sentence'] = doc['sentence'].str.strip()
        #doc = doc[doc['sentence'] != '']
        doc.to_json(save_path + '/' + text_files[i_file].replace('.txt', '.json'), orient='records', indent=4, force_ascii=False)
        txt_file_path = save_path + '/' + text_files[i_file]
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            for sentence in doc['sentence']:
                f.write(sentence + "\n")

def unify_text_and_labels_multiclass_LLM(texts_path, labels_path, save_path):
    # TO DO: optimize (remove hierarchy calculation)
    """Unify text and labels into a single JSON file.
    The labelling approach is a multiclass {BIO}.
    It also generates new .txt files with the treatment that was applied to the text.

    Args:
        texts_path (str): Path to the directory containing the text files.
        labels_path (str): Path to the directory containing the label files.
        save_path (str): Path to the directory where the unified json files will be saved.
    """
    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')
    for i_file in range(len(label_files)):
        dict_label_range = {}
        with open(labels_path + '/' + label_files[i_file], 'r') as file:
            file_content = json.load(file)
            for block in file_content:
                label = block['block_type']
                if label in dict_label_range:
                    dict_label_range[label].append((block['start_line'] - 1, block['end_line'] - 1))
                else:
                    dict_label_range[label] = [(block['start_line'] - 1, block['end_line'] - 1)]
        value_hierarchie_single = [{tup: 1} for sublist in dict_label_range.values() for tup in sublist]
        for i in range(len(value_hierarchie_single)):
            current_value = value_hierarchie_single[i]
            current_start, current_end = list(current_value.keys())[0]
            for j in range(len(value_hierarchie_single)):
                if i == j:
                    continue
                potential_value = value_hierarchie_single[j]
                potential_start, potential_end = list(potential_value.keys())[0]
                if (current_start >= potential_start) and (current_end <= potential_end):
                    value_hierarchie_single[i][(current_start, current_end)] += 1
        
        doc = pd.DataFrame(columns=['sentence', 'label'])
        with open(texts_path + '/' + text_files[i_file], 'r', encoding="utf-8-sig", errors='ignore') as file:
            lines = file.readlines()
            lines = [clean_line(line) for line in lines]
            line_id = 0
            for line in lines:
                current_max = 0
                current_interval = (0, 0)
                for i in range(len(value_hierarchie_single)):
                    current_value = value_hierarchie_single[i]
                    current_start, current_end = list(current_value.keys())[0]
                    if (line_id >= current_start) and (line_id <= current_end):
                        if value_hierarchie_single[i][(current_start, current_end)] >= current_max:
                            current_max = value_hierarchie_single[i][(current_start, current_end)]
                            current_interval = (current_start, current_end)
                for label in dict_label_range:
                    if current_interval in dict_label_range[label]:
                        current_label = label
                        if current_label == 'Section':
                            current_label = 'Article'
                if current_max == 0 or current_label == 'Other':
                    doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': 'O'})], ignore_index=True)
                else:
                    if line_id == current_interval[0]:
                        doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'B-{current_label}'})], ignore_index=True)
                    else:
                        doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'I-{current_label}'})], ignore_index=True)
                line_id +=1
        doc['sentence'] = doc['sentence'].str.strip()
        #doc = doc[doc['sentence'] != '']
        doc.to_json(save_path + '/' + text_files[i_file].replace('.txt', '.json'), orient='records', indent=4, force_ascii=False)
        txt_file_path = save_path + '/' + text_files[i_file]
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            #line_id = 0
            for sentence in doc['sentence']:
            #    line_id += 1
            #    f.write("{" + f"{line_id}" + "} : " + sentence + "\n")
                f.write(sentence + "\n")

def generate_html_txt_files(texts_path, labels_path, save_path):
    """Generate the contract txt files simulating the HTML format with the labels.

    Args:
        texts_path (str): Path to the directory containing the text files.
        labels_path (str): Path to the directory containing the label files.
        save_path (str): Path to the directory where the unified json files will be saved.
    """
    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')
    for i_file in range(len(label_files)):
        with open(labels_path + '/' + label_files[i_file], 'r') as label_file:
            file_content = json.load(label_file)
            start_lines = {}
            end_lines = {}
            for block in file_content:
                if block['start_line'] - 1 in start_lines:
                    start_lines[block['start_line'] - 1].append(block['block_type'])
                else:
                    start_lines[block['start_line'] - 1] = [block['block_type']]
                if block['end_line'] - 1 in end_lines:
                    end_lines[block['end_line'] - 1].append(block['block_type'])
                else:
                    end_lines[block['end_line'] - 1] = [block['block_type']]
        with open(texts_path + '/' + text_files[i_file], 'r', encoding="utf-8-sig", errors='ignore') as text_file:
            html_txt = ""
            lines = text_file.readlines()
            lines = [clean_line(line) for line in lines]
            line_id = 0
            for line in lines:
                if line_id in start_lines:
                    _start = [f"<{item}>" for item in start_lines[line_id]]
                    for i in range(len(_start)):
                        if _start[i] == "<Section>":
                            _start[i] = "<Article>"
                    html_txt += "".join(_start)
                html_txt += line
                if line_id in end_lines:
                    _end = [f"</{item}>" for item in end_lines[line_id]]
                    for i in range(len(_end)):
                        if _end[i] == "</Section>":
                            _end[i] = "</Article>"
                    html_txt += "".join(_end)
                html_txt += "\n"
                line_id+=1
        with open(save_path + '/' + text_files[i_file], 'w', encoding='utf-8-sig') as html_file:
            html_file.write(html_txt)
            html_file.close()

def generate_html_from_unified_json_v2(unified_json_path, save_path):
    # Read the unified JSON file
    label_files = get_files(unified_json_path, 'json')
    for i_file in range(len(label_files)):
        with open(f'{unified_json_path}/{label_files[i_file]}', 'r', encoding="utf-8-sig", errors='ignore') as json_file:
            data = json.load(json_file)
        html_txt = ""
        line_id = 0
        for entry in data:
            line_id += 1
            sentence = entry.get("sentence", "")
            labels = entry.get("labels", [])
            
            # Start building the output line
            line = "{" + f"{line_id}" + "} "
            begin_labels = []
            normal_labels = []

            for label in labels:
                if label[0] == "B":
                    begin_labels.append(label)
                else:
                    normal_labels.append(label)
            
            for normal_label in normal_labels:
                line += "{" + normal_label + "}"
            if len(begin_labels) > 0:
                line += " "
            for begin_label in begin_labels:
                line+= f"({begin_label})"
            line+= " : "

            line += sentence
            # Append the processed line followed by a newline
            html_txt += line + "\n"
    
        # Write the generated text to the save file
        with open(f'{save_path}/' + label_files[i_file].replace('json', 'txt'), 'w', encoding='utf-8-sig') as output_file:
            output_file.write(html_txt)
