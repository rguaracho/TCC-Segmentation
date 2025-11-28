import re
import json
import os

def process_files(metadata_file_path, original_text_file_path):
    # Load the sentences JSON file (expects a list of objects with a "sentence" field)
    with open(original_text_file_path, 'r', encoding='utf-8') as orig_file:
        original_sentences = [line.strip() for line in orig_file]

    results = []
    
    # Regular expression explanation:
    # ^\{(\d+)\}                : Captures the line id (digits inside {}) at the start.
    # \s*\{([^}]+)\}             : Captures the label from inside the next {}.
    # (?:\s*\((Begin)[^)]*\))?    : Optionally captures the token "Begin" if present.
    # :                          : Matches the literal colon.
    pattern = re.compile(r'^\{(\d+)\}\.?\s*\{([^}]+?)\}(?:\s*\((Begin(?:\s+[^)]*)?)\))?\s*:')

    # Process each metadata line
    with open(metadata_file_path, 'r', encoding='utf-8') as meta_file:
        for meta_line in meta_file:
            #print(meta_line)
            meta_line = meta_line.strip()
            #print(meta_line)
            #if not meta_line:
            #    continue  # Skip empty lines

            match = pattern.match(meta_line)
            #print(match)
            if match:
                line_id = int(match.group(1))
                label_text = match.group(2)
                begin_flag = match.group(3)
                
                # Prefix the label with "B-" if "Begin" flag is found, otherwise with "I-"
                prefix = "B-" if begin_flag else "I-"
                full_label = prefix + label_text

                # Use the line_id (1-based indexing) to find the matching sentence in the JSON file
                if 1 <= line_id <= len(original_sentences):
                    sentence = original_sentences[line_id - 1]
                else:
                    print(f"Warning: Line id {line_id} is out of range in the sentences JSON file.")
                    sentence = ""
                
                results.append({
                    "sentence": sentence,
                    "label": full_label
                })
            else:
                print(f"Warning: Could not parse metadata line: {meta_line}")
    
    return results
    
    # Save the resulting JSON to the output file path
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        json.dump(results, out_file, indent=4, ensure_ascii=False)
    
    print(f"JSON successfully saved to {output_file_path}")

def process_file(metadata_file_path, original_text_file_path, output_file_path):
    # Load the sentences JSON file (expects a list of objects with a "sentence" field)
    with open(original_text_file_path, 'r', encoding='utf-8') as orig_file:
        original_sentences = [line.strip() for line in orig_file]
    with open(metadata_file_path, 'r', encoding='utf-8') as meta_file:
        meta_lines = meta_file.readlines()

    if (len(original_sentences) != len(meta_lines)):
        print(f"ERROR FOR DOC: {metadata_file_path}")
        print(f"DOCUMENTS WITHOUT SAME LEN {len(original_sentences)} != {len(meta_lines)}")

    results = []
    exhibit = False
    if meta_lines[0][2] == "1" and "Begin Exhibit" in meta_lines[0]:
        exhibit = True
    
    for meta_line, original_line in zip(meta_lines, original_sentences):
        #print(meta_line)
        meta_line = meta_line.strip()
        #print(meta_line)
        #if not meta_line:
        #    continue  # Skip empty lines
        match = re.search(":", meta_line)
        #print(match)
        if match:
            labels = meta_line[:match.start()]
            labels = re.findall(r'[{\(]([^}\)]+)[}\)]', labels)
            #print(labels)
        else:
            print("ERROR: cant find match in line")
            print(meta_line)
            return
        line_labels = []
        begin_label = False
        for label in labels:
            if 'Begin' in label:
                if 'Begin' not in line_labels:
                    line_labels.append('Begin')
                    begin_label = True
            elif label.isdigit() == False:
                line_labels.append(label)
        if exhibit and 'Exhibit' not in labels:
            line_labels.append('Exhibit')
        
        if labels != []:
            correct_line = original_line[len("{" + f"{labels[0]}" + "} : "):]
        else:
            correct_line = original_line
        results.append({
                    "sentence": correct_line,
                    "labels": line_labels
                })
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        json.dump(results, out_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    """
    LLM_RESPONSE_PATH = "dataset/llm_response/"
    ORIGINAL_PATH = "dataset/unified_LLM/"
    SAVE_PATH = "dataset/LLM_output/"
    split_paths = "dataset/LLM_split_paths_merge.json"
    with open(split_paths, 'r') as f:
        config = json.load(f)
    test_files = config.get("test", [])
    contracts = [os.path.basename(file_path)[:-5] for file_path in test_files]
    """

    LLM_RESPONSE_PATH = "dataset/llm_response/"
    ORIGINAL_PATH = "dataset/original_unified_LLM/"
    SAVE_PATH = "dataset/last_4o_mini/"
    split_paths = "dataset/LLM_split_paths.json"
    #all_items = sorted(os.listdir(ORIGINAL_PATH))
    #test_file_names = [file_path[:-4] for file_path in all_items]
    with open(split_paths, 'r') as f:
        config = json.load(f)
    test_files = config.get("test", [])
    contracts = [os.path.basename(file_path)[:-5] for file_path in test_files]
    #contracts = test_file_names
    for contract in contracts:
        process_file(f"{LLM_RESPONSE_PATH}{contract}.txt", f"{ORIGINAL_PATH}{contract}.txt", f"{SAVE_PATH}{contract}.json")
