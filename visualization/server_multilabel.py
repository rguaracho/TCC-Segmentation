from http.server import SimpleHTTPRequestHandler, HTTPServer 
import json
import random

# Global dictionary to store normalized label-to-color mappings (unused in indentation version)
LABEL_COLORS = {}

# Default paths for predictions and ground truths.
PREDICTIONS_DIR = "../predictions/test"
GROUND_TRUTHS_DIR = "../dataset/unified"

def normalize_label(label):
    """
    Remove the prefix "B-" or "I-" from a label if present.
    For example, "B-Introduction of Parties" becomes "Introduction of Parties".
    """
    if label.startswith("B-") or label.startswith("I-"):
        return label.split('-', 1)[1].strip()
    return label

def get_offset_margin(labels_list, base_offset=20):
    """
    Compute a left margin offset (in pixels) based on the number of labels
    in the list that are not 'Begin' (case-insensitive).
    Each such label adds 'base_offset' pixels.
    """
    non_begin = [lab for lab in labels_list if lab.lower() != "begin"]
    return len(non_begin) * base_offset

def should_start_new_segment(prev_labels, curr_labels):
    """
    For multilabel segmentation, start a new segment if the current item's labels list:
      - Contains "Begin"
      - And is different from the previous item's labels.
    Both prev_labels and curr_labels are expected to be lists.
    """
    if "Begin" in curr_labels and curr_labels != prev_labels:
        return True
    return False

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # --- Predictions view endpoint for multilabel ---
        if self.path.startswith("/singleview/predictions/"):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            try:
                prefix = "/singleview/predictions/"
                file_path_relative = self.path[len(prefix):]
                file_path = f'{PREDICTIONS_DIR}/{file_path_relative}'
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                # Expect each item to have a "predicted_labels" field (a list).
                segments = []
                if data:
                    current_segment = [data[0]]
                    prev_labels = data[0].get("predicted_labels", data[0].get("label", []))
                    if not isinstance(prev_labels, list):
                        prev_labels = [prev_labels]
                    for item in data[1:]:
                        curr_labels = item.get("predicted_labels", item.get("label", []))
                        if not isinstance(curr_labels, list):
                            curr_labels = [curr_labels]
                        if should_start_new_segment(prev_labels, curr_labels):
                            segments.append(current_segment)
                            current_segment = [item]
                        else:
                            current_segment.append(item)
                        prev_labels = curr_labels
                    segments.append(current_segment)
                else:
                    segments = []
                
                html_content = "<html><body>"
                for segment in segments:
                    html_content += '<div style="border: 1px solid #ddd; padding: 5px; margin:5px 0;">'
                    for item in segment:
                        sentence = item["sentence"]
                        labels_list = item.get("predicted_labels", item.get("label", []))
                        if not isinstance(labels_list, list):
                            labels_list = [labels_list]
                        # Compute margin for this line based on its labels (excluding "Begin")
                        margin = get_offset_margin(labels_list, base_offset=20)
                        labels_str = ", ".join(labels_list)
                        html_content += (
                            f'<div style="margin-left: {margin}px;">'
                            f'<p style="margin:0;">{sentence} '
                            f'<span style="font-size: 0.8em; color: #333;">({labels_str})</span></p>'
                            f'</div>'
                        )
                    html_content += '</div>'
                html_content += "</body></html>"
                self.wfile.write(html_content.encode('utf-8'))
            except FileNotFoundError:
                error_message = "<html><body><h1>Predictions file not found</h1></body></html>"
                self.wfile.write(error_message.encode('utf-8'))
            except json.JSONDecodeError:
                error_message = "<html><body><h1>Error decoding JSON in predictions file</h1></body></html>"
                self.wfile.write(error_message.encode('utf-8'))
            except Exception as e:
                error_message = f"<html><body><h1>{str(e)}</h1></body></html>"
                self.wfile.write(error_message.encode('utf-8'))
        
        # --- Unified view endpoint for multiview (ground truth) ---
        elif self.path.startswith("/multiview/unified/"):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            try:
                prefix = "/multiview/unified/"
                filename = self.path[len(prefix):]
                file_path = f'{GROUND_TRUTHS_DIR}/{filename}'
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                segments = []
                if data:
                    current_segment = [data[0]]
                    prev_labels = data[0].get("labels", [])
                    if not isinstance(prev_labels, list):
                        prev_labels = [prev_labels]
                    for item in data[1:]:
                        curr_labels = item.get("labels", [])
                        if not isinstance(curr_labels, list):
                            curr_labels = [curr_labels]
                        if should_start_new_segment(prev_labels, curr_labels):
                            segments.append(current_segment)
                            current_segment = [item]
                        else:
                            current_segment.append(item)
                        prev_labels = curr_labels
                    segments.append(current_segment)
                else:
                    segments = []
                
                html_content = "<html><body>"
                for segment in segments:
                    html_content += '<div style="border: 1px solid #ddd; padding: 5px; margin:5px 0;">'
                    for item in segment:
                        sentence = item["sentence"]
                        labels_list = item.get("labels", [])
                        if not isinstance(labels_list, list):
                            labels_list = [labels_list]
                        margin = get_offset_margin(labels_list, base_offset=20)
                        labels_str = ", ".join(labels_list)
                        html_content += (
                            f'<div style="margin-left: {margin}px;">'
                            f'<p style="margin:0;">{sentence} '
                            f'<span style="font-size: 0.8em; color: #333;">({labels_str})</span></p>'
                            f'</div>'
                        )
                    html_content += '</div>'
                html_content += "</body></html>"
                self.wfile.write(html_content.encode('utf-8'))
            except FileNotFoundError:
                error_message = "<html><body><h1>Unified file not found</h1></body></html>"
                self.wfile.write(error_message.encode('utf-8'))
            except json.JSONDecodeError:
                error_message = "<html><body><h1>Error decoding JSON in unified file</h1></body></html>"
                self.wfile.write(error_message.encode('utf-8'))
            except Exception as e:
                error_message = f"<html><body><h1>{str(e)}</h1></body></html>"
                self.wfile.write(error_message.encode('utf-8'))
        
        # --- Serve index_multilabel.html ---
        elif self.path == '/' or self.path == '/index_multilabel.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            try:
                with open('index_multilabel.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    self.wfile.write(html_content.encode('utf-8'))
            except FileNotFoundError:
                error_message = "<html><body><h1>index_multilabel.html not found</h1></body></html>"
                self.wfile.write(error_message.encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>File not found</h1></body></html>")

def run(server_class=HTTPServer, handler_class=MyHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Serving on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
