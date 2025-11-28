import tiktoken
import json
import math
from LLM_retriever_window import Retriever

# Configuration Parameters
FEW_SHOT_PERCENTAGE = 0
TOTAL_TOKENS = 32000

FEW_SHOT_TOKENS = TOTAL_TOKENS * FEW_SHOT_PERCENTAGE # 400
MAIN_TASK_TOKENS = TOTAL_TOKENS - FEW_SHOT_TOKENS # 1600

MAX_TOKENS_PER_SENTENCE = 35 #couper plus le few shot, 2 max separés, garder le percentile 
SLIDING_WINDOW_SIZE = 5

INCLUDE_INSTRUCTIONS = True
OUTPUT_FORMAT = "JSON"  # or "TEXT"
LABELS = ["Preamble", "Article", "Section", "Reference Number/Date", 
          "Contract Title", "Introduction of Parties", "Signature", 
          "Exhibit", "Table of Contents", "Other"]

INSTRUCTIONS = (
    f"""You are a sequential classifier for legal documents. You have the following possible labels: [{", ".join(LABELS)}].
We tag our sequences following the criteria: B/I - label - hierarchy level, as contracts may contain nested sections.
Here are some correct examples: \n"""
)

MISSION = (
    f"""Now, classify the following sequential sequences, providing the output imitating a HTML format \n"""
)

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(encoding.encode(text))

def split_sentences_to_be_classified(sentences):
    current_tokens = 0
    index = 0
    sentences_to_be_classified = []
    while current_tokens < MAIN_TASK_TOKENS and index < len(sentences):
        sentence = sentences[index]
        tokens = encoding.encode(sentence)
        sentences_to_be_classified.append(sentence)
        current_tokens += len(tokens)
        index+=1
    return sentences_to_be_classified

def split_sentences_by_token_limit(sentences, max_tokens):
    split_sentences = []
    for sentence in sentences:
        tokens = encoding.encode(sentence)
        if len(tokens) <= max_tokens:
            split_sentences.append(sentence)
        else:
            truncated_tokens = tokens[:max_tokens]
            truncated_sentence = encoding.decode(truncated_tokens)
            split_sentences.append(truncated_sentence)
    return split_sentences

def generate_few_shot_examples(data, num_examples):
    few_shot = data[:num_examples]
    examples = ""
    for example in few_shot:
        examples += f"Input: {example['sentence']}\nOutput: {example['label']}\n\n"
    return examples

def generate_main_task_prompt(data, start_index, window_size):
    window = data[start_index:start_index + window_size]
    prompt = ""
    for example in window:
        prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
    return prompt

#def create_prompt(data, data_to_be_classified, include_instructions=True):
def create_prompt(data_to_be_classified, include_instructions=True):
    '''
    final_prompt = ''

    sentences = [sentence["sentence"] for sentence in data]
    number = math.ceil(FEW_SHOT_TOKENS/MAX_TOKENS_PER_SENTENCE)
    sentences = split_sentences_by_token_limit(sentences[:number], MAX_TOKENS_PER_SENTENCE)
    labels = [sentence["label"] for sentence in data][:number]
    final_prompt += INSTRUCTIONS
    prompts = []
    for i in range(len(sentences)):
        prompt = f"{sentences[i]} / Label: {labels[i]} \n"
        prompts.append(prompt)
    final_prompt += " ".join(prompts)

    final_prompt += MISSION
    '''

    sentences_to_be_classified = [sentence["sentence"] for sentence in data_to_be_classified]
    sentences_to_be_classified = split_sentences_to_be_classified(sentences_to_be_classified)
    prompts = []
    final_prompt = ""
    for i in range(len(sentences_to_be_classified)):
        prompt = f"{sentences_to_be_classified[i]}"
        prompts.append(prompt)
    print("doc len: ", len(prompts))
    return prompts
    final_prompt += "\n".join(prompts)
    #print(final_prompt)
    print(count_tokens(final_prompt))
    return final_prompt
    
    # Implement sliding window
    prompts = []
    for i in range(0, len(sentences), SLIDING_WINDOW_SIZE):
        window = sentences[i:i + SLIDING_WINDOW_SIZE]
        window_text = " ".join(window)
        if include_instructions:
            instructions = "Please classify the following examples:\n"
        else:
            instructions = ""
        prompt = f"{instructions}{few_shot_examples}\n{window_text}\nOutput:"
        prompts.append(prompt)
    
    return prompts

def get_instructions_prompt():

    #maybe add something asking for focus in bigger sections instead of nested sections

    instructions = "\n".join([
        "You are a legal text classifier. Your task is to: ",
        "- Read each line of the provided text.",
        "- Assign one or more of the following labels to each line: [Contract Title, Article, Introduction of Parties, Table of Contents, Preamble, Signature, Exhibit, Reference Number/Date].",
        "- For every label that applies to a line, include it in the output.",
        "- If a line marks the start of a 'new top-level section' for any label (even if nested within another label), prepend that line with '(Begin {label})'.",
        "- A 'new top-level section' refers to the changement of a label from the previous the line",
        "- Do not add 'Begin Article' for sub-articles (e.g., 1.1, 1.2, etc.). Treat these lines as continuations of the same Article.",
        "- Continue applying all the 'top-level sections' labels to subsequent lines.",
        "- Output in the specified format.",

        #"You should provide the output with XML tags, you should not mix the content of the tags with the text. Examples: <Article>, <Contract Title>.",
        #"You should not aggregate NOR split lines in your answer.",
        #"You should use lines outlined by [CONTEXT] [/CONTEXT] as context for next classifications, do not repeat or output them.",
        #"If you dont receive a [CONTEXT], you are in the beginning of a document. Usually <Contract Title>, <Introduction of Parties>, <Preamble> and <Table of Contents> are there.",
        #"Provide the output with exactly the same length of lines of the input text.",
        #"You should not create new tags nor labels.",
        #"You should classify nested contests, for example, you can have article inside article, or article inside exhibits.."
        #"If there is a continuous tag after closing in the context, example: </Article> [CONTINUOUS], you should keep the classification for next lines."
        #"Provide the output with exactly the same length of lines of the input text."
    ])
    return instructions


def get_context_prompts():
    context_prompts = []
    for i in [0,5]:
        with open(f'dataset/unified_LLM/contract{i}.txt', 'r', encoding='utf-8') as file:
            user_content = file.read()
        with open(f'dataset/html_txts/contract{i}.txt', 'r', encoding='utf-8') as file:
            assistant_content = file.read().lstrip('\ufeff')
        context_prompts.append((user_content, assistant_content))
    return context_prompts


def get_prompts(retriever: Retriever, query: str, top_k: int):
    '''
    with open(f'dataset/unified_LLM/{contract_basename}.json', 'r', encoding='utf-8') as file:
        data_to_be_classified = json.load(file)
    prompt = create_prompt(data_to_be_classified)
    '''
    input_example = "Example:\nINPUT:\n"
    output_example = "Expected OUTPUT:\n"
    context_prompts = []
    for labeled_txt, original_txt, _ in retriever.get_context(query_lines=query.split("\n"), top_k=top_k):
        original = input_example + original_txt
        labeled = output_example + labeled_txt
        context_prompts.append((original, labeled))

    #original, label = [(txt, doc) for doc, txt, _ in retriever.get_context(query_lines=query.split("\n"), top_k=top_k)]
    #context_prompts = [(txt, doc) for doc, txt, _ in retriever.get_context(query_lines=query.split("\n"), top_k=top_k)]
    #user_prompt = "\n".join(prompt)
    user_prompt = []
    return {"context_prompts": context_prompts, "user_prompt": user_prompt}


def main():
    #retriever = Retriever.load("retriever/retriever_state.pkl")
    #print(get_prompts(retriever, 10))
    #return [f for f in all_files if f in train_file_names]

    retriever = Retriever.load("retriever/retriever_state.pkl")
    query_lines = "CONTRAT DE CONDITIONS DE LIVRAISON STANDARD\n Numéro de Contrat :\n les Le présent Contrat, constitué des conditions généraet des conditions particulières, est établi entre le Client et le Gestionnaire du Réseau de Distribution Gaz (GRD) désignés ci-après :"
         
    B = get_prompts(retriever, query_lines, 1)
    print(B)

    print(5)


if __name__ == "__main__":
    main()
