import pickle
import os
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import json
import re

class Retriever:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self.labeled_texts = None
        self.original_texts = None
        self.embeddings = None
        
    @staticmethod
    def _read_txts(folder_path, split_dict):
        paths = json.load(open(split_dict, "r", errors="ignore"))['train']
        paths = [path.replace("dataset/unified", folder_path) for path in paths]
        paths = [path.replace("json", "txt") for path in paths]
        documents = []
        for path in paths:
            with open(path, "r", encoding="utf-8-sig", errors="ignore") as file:
                lines = file.readlines()
                documents.append(lines)
        return documents
    
    @staticmethod
    def _truncate_lines(original_lines, labeled_lines, max_text_length):
        def truncate_line(line, max_len):
            match = re.search(":", line)
            if line.endswith('\n'):
                line = line[:-1]
                line = line[:match.start() + max_len]
                line += '\n'
            else:
                line = line[:match.start() + max_len]

        truncated_original_lines = []
        truncated_labeled_lines = []

        for i in range(len(labeled_lines)):
            if len(labeled_lines[i]) > max_text_length:
                truncated_labeled_lines.append(truncate_line(labeled_lines[i], max_text_length))
                truncated_original_lines.append(truncate_line(original_lines[i], max_text_length))
                continue
            truncated_labeled_lines.append(labeled_lines[i])
            truncated_original_lines.append(original_lines[i])

        return truncated_original_lines, truncated_labeled_lines
    
    def generate_embeddings(self, texts_folder, labeled_texts_folder, split_dict, window_size=200, overlap_percent=70, max_line_length=100, save=True):
        text_files = self._read_txts(texts_folder, split_dict)
        labeled_text_files = self._read_txts(labeled_texts_folder, split_dict)
        
        all_labeled_text_chunks = []
        all_original_text_chunks = []
        all_segment_embeddings = []

        step = int(window_size * (1 - overlap_percent / 100))
        if step <= 0:
            raise ValueError("overlap_percent must be less than 100")

        text_indices = tqdm(range(len(text_files)), desc="Generating embeddings", unit="document")
        for i in text_indices:
            original_lines = text_files[i]
            labeled_lines = labeled_text_files[i]
            num_lines = len(original_lines)
            
            if num_lines <= window_size:
                starts = [0]
            else:
                # generate start indices ensuring a full window can be taken
                starts = list(range(0, num_lines - window_size + 1, step))
                # if the last window doesn't end exactly at the end, add one more window starting at the last possible index
                if starts[-1] != num_lines - window_size:
                    starts.append(num_lines - window_size)

            for start in starts:
                end = start + window_size
                original_window = original_lines[start:end]
                labeled_window = labeled_lines[start:end]

                if not labeled_window:
                    continue
                
                # generate embeddings for the labeled window and average them to get a single embedding
                window_line_embeddings = self.model.encode(labeled_window, convert_to_tensor=True)
                window_embedding = torch.mean(window_line_embeddings, dim=0)
                
                # crop excessively long lines
                cropped_original_window, cropped_labeled_window = self._truncate_lines(original_window, labeled_window, max_line_length)
                
                original_window_text = "\n".join(cropped_original_window)
                labeled_window_text = "\n".join(cropped_labeled_window)
                #print(len(labeled_window_text))
                all_original_text_chunks.append(original_window_text)
                all_labeled_text_chunks.append(labeled_window_text)
                all_segment_embeddings.append(window_embedding)
        
        self.labeled_texts = all_labeled_text_chunks
        self.original_texts = all_original_text_chunks
        self.embeddings = torch.stack(all_segment_embeddings) if all_segment_embeddings else None
        
        if save:
            self.save("retriever/retriever_state.pkl")
        
    def get_context(self, query_lines, top_k=1):
        # Generate embeddings for each query line
        line_embeddings = self.model.encode(query_lines, convert_to_tensor=True)
        # Average the embeddings to produce a single query-level embedding
        query_embedding = torch.mean(line_embeddings, dim=0, keepdim=True)
        # Compute cosine similarity between the averaged query embedding and document embeddings
        cosine_scores = util.cos_sim(query_embedding, self.embeddings.to(self.device))[0]
        # Get the indices of the top_k most similar documents
        top_results = torch.topk(cosine_scores, k=top_k)
        results = [(self.labeled_texts[idx], self.original_texts[idx], score.item()) for score, idx in zip(top_results[0], top_results[1])]
        return results

    def save(self, file_path):
        """
        Saves the retriever's state (documents, model name, and precomputed embeddings) to a file.
        
        Args:
            file_path (str): The file path to save the state.
        """
        data = {
            'labeled_texts': self.labeled_texts,
            'original_texts': self.original_texts,
            'embeddings': self.embeddings.cpu().numpy(),
            'model_name': self.model_name
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Retriever state saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        """
        Loads the retriever's state from a file without recomputing embeddings.
        
        Args:
            file_path (str): The file path where the state is saved.
        
        Returns:
            Retriever: An instance with the loaded state.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # Create an instance with the stored model name
        instance = cls(model_name=data['model_name'])
        # Load the documents and embeddings from the saved state
        instance.labeled_texts = data['labeled_texts']
        instance.original_texts = data['original_texts']
        instance.embeddings = torch.tensor(data['embeddings'])
        print(f"Retriever state loaded from {file_path}")
        return instance