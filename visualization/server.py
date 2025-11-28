from http.server import SimpleHTTPRequestHandler, HTTPServer
import json
import random

# Global dictionary to store normalized label-to-color mappings
LABEL_COLORS = {}

# Default paths for predictions and ground truths (used for original endpoints).
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

def get_rgba_color(label, alpha=0.6):
    """
    Return a consistent rgba color string for a given label.
    The label is normalized (i.e. "B-Intro" and "I-Intro" both become "Intro").
    """
    normalized = normalize_label(label)
    if normalized not in LABEL_COLORS:
        hex_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b_val = int(hex_color[5:7], 16)
        LABEL_COLORS[normalized] = (r, g, b_val)
    else:
        r, g, b_val = LABEL_COLORS[normalized]
    return f"rgba({r}, {g}, {b_val}, {alpha})"

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # --- Original Predictions view endpoint ---
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

                # Use 'predicted_label' if available; fallback to 'label'
                is_bio = False
                if data:
                    first_label = data[0].get("predicted_label", data[0].get("label", ""))
                    if first_label.startswith("B-") or first_label.startswith("I-"):
                        is_bio = True

                if is_bio:
                    def should_start_new_segment(prev_label, curr_label):
                        if curr_label.startswith("B"):
                            return True
                        if curr_label == "O" and prev_label != "O":
                            return True
                        if curr_label.startswith("I") and (prev_label is None or prev_label == "O"):
                            return True
                        if curr_label.startswith("I") and prev_label is not None and prev_label != "O":
                            if '-' in curr_label and '-' in prev_label:
                                return curr_label.split('-', 1)[1] != prev_label.split('-', 1)[1]
                            else:
                                return curr_label != prev_label
                        return False
                else:
                    def should_start_new_segment(prev_label, curr_label):
                        return prev_label is not None and curr_label != prev_label

                segments = []
                if data:
                    current_segment = [data[0]]
                    prev_label = data[0].get("predicted_label", data[0].get("label", ""))
                    for item in data[1:]:
                        curr_label = item.get("predicted_label", item.get("label", ""))
                        if should_start_new_segment(prev_label, curr_label):
                            segments.append(current_segment)
                            current_segment = [item]
                        else:
                            current_segment.append(item)
                        prev_label = curr_label
                    segments.append(current_segment)
                else:
                    segments = []

                html_content = "<html><body>"
                for segment in segments:
                    segment_label = segment[0].get("predicted_label", segment[0].get("label", ""))
                    rgba_color = get_rgba_color(segment_label, 0.6)
                    html_content += f'<div style="background-color:{rgba_color}; padding: 5px; margin:5px 0;">'
                    for item in segment:
                        sentence = item["sentence"]
                        label_used = item.get("predicted_label", item.get("label", ""))
                        html_content += (
                            f'<p style="margin:0;">{sentence} '
                            f'<span style="font-size: 0.8em; color: #333;">({label_used})</span></p>'
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

        # --- Custom Predictions view endpoint ---
        elif self.path.startswith("/..dataset/LLM_output/"):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            try:
                prefix = "/..dataset/LLM_output/"
                file_path_relative = self.path[len(prefix):]
                file_path = f'../dataset/LLM_output/{file_path_relative}'
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # Use 'predicted_label' if available; fallback to 'label'
                is_bio = False
                if data:
                    first_label = data[0].get("predicted_label", data[0].get("label", ""))
                    if first_label.startswith("B-") or first_label.startswith("I-"):
                        is_bio = True

                if is_bio:
                    def should_start_new_segment(prev_label, curr_label):
                        if curr_label.startswith("B"):
                            return True
                        if curr_label == "O" and prev_label != "O":
                            return True
                        if curr_label.startswith("I") and (prev_label is None or prev_label == "O"):
                            return True
                        if curr_label.startswith("I") and prev_label is not None and prev_label != "O":
                            if '-' in curr_label and '-' in prev_label:
                                return curr_label.split('-', 1)[1] != prev_label.split('-', 1)[1]
                            else:
                                return curr_label != prev_label
                        return False
                else:
                    def should_start_new_segment(prev_label, curr_label):
                        return prev_label is not None and curr_label != prev_label

                segments = []
                if data:
                    current_segment = [data[0]]
                    prev_label = data[0].get("predicted_label", data[0].get("label", ""))
                    for item in data[1:]:
                        curr_label = item.get("predicted_label", item.get("label", ""))
                        if should_start_new_segment(prev_label, curr_label):
                            segments.append(current_segment)
                            current_segment = [item]
                        else:
                            current_segment.append(item)
                        prev_label = curr_label
                    segments.append(current_segment)
                else:
                    segments = []

                html_content = "<html><body>"
                for segment in segments:
                    segment_label = segment[0].get("predicted_label", segment[0].get("label", ""))
                    rgba_color = get_rgba_color(segment_label, 0.6)
                    html_content += f'<div style="background-color:{rgba_color}; padding: 5px; margin:5px 0;">'
                    for item in segment:
                        sentence = item["sentence"]
                        label_used = item.get("predicted_label", item.get("label", ""))
                        html_content += (
                            f'<p style="margin:0;">{sentence} '
                            f'<span style="font-size: 0.8em; color: #333;">({label_used})</span></p>'
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

        # --- Original Unified view endpoint for multiview ---
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

                # Auto-detect the scheme using the first element's "label"
                is_bio = False
                if data:
                    first_label = data[0].get("label", "")
                    if first_label.startswith("B-") or first_label.startswith("I-"):
                        is_bio = True

                if is_bio:
                    def should_start_new_segment(prev_label, curr_label):
                        if curr_label.startswith("B"):
                            return True
                        if curr_label == "O" and prev_label != "O":
                            return True
                        if curr_label.startswith("I") and (prev_label is None or prev_label == "O"):
                            return True
                        if curr_label.startswith("I") and prev_label is not None and prev_label != "O":
                            if '-' in curr_label and '-' in prev_label:
                                return curr_label.split('-', 1)[1] != prev_label.split('-', 1)[1]
                            else:
                                return curr_label != prev_label
                        return False
                else:
                    def should_start_new_segment(prev_label, curr_label):
                        return prev_label is not None and curr_label != prev_label

                segments = []
                if data:
                    current_segment = [data[0]]
                    prev_label = data[0]["label"]
                    for item in data[1:]:
                        curr_label = item["label"]
                        if should_start_new_segment(prev_label, curr_label):
                            segments.append(current_segment)
                            current_segment = [item]
                        else:
                            current_segment.append(item)
                        prev_label = curr_label
                    segments.append(current_segment)
                else:
                    segments = []

                html_content = "<html><body>"
                for segment in segments:
                    segment_label = segment[0]["label"]
                    rgba_color = get_rgba_color(segment_label, 0.6)
                    html_content += f'<div style="background-color:{rgba_color}; padding: 5px; margin:5px 0;">'
                    for item in segment:
                        sentence = item["sentence"]
                        label = item["label"]
                        html_content += (
                            f'<p style="margin:0;">{sentence} '
                            f'<span style="font-size: 0.8em; color: #333;">({label})</span></p>'
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

        # --- Custom Unified view endpoint ---
        elif self.path.startswith("/..dataset/unified_LLM/"):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            try:
                prefix = "/..dataset/unified_LLM/"
                file_path_relative = self.path[len(prefix):]
                file_path = f'../dataset/unified_LLM/{file_path_relative}'
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # Auto-detect the scheme using the first element's "label"
                is_bio = False
                if data:
                    first_label = data[0].get("label", "")
                    if first_label.startswith("B-") or first_label.startswith("I-"):
                        is_bio = True

                if is_bio:
                    def should_start_new_segment(prev_label, curr_label):
                        if curr_label.startswith("B"):
                            return True
                        if curr_label == "O" and prev_label != "O":
                            return True
                        if curr_label.startswith("I") and (prev_label is None or prev_label == "O"):
                            return True
                        if curr_label.startswith("I") and prev_label is not None and prev_label != "O":
                            if '-' in curr_label and '-' in prev_label:
                                return curr_label.split('-', 1)[1] != prev_label.split('-', 1)[1]
                            else:
                                return curr_label != prev_label
                        return False
                else:
                    def should_start_new_segment(prev_label, curr_label):
                        return prev_label is not None and curr_label != prev_label

                segments = []
                if data:
                    current_segment = [data[0]]
                    prev_label = data[0]["label"]
                    for item in data[1:]:
                        curr_label = item["label"]
                        if should_start_new_segment(prev_label, curr_label):
                            segments.append(current_segment)
                            current_segment = [item]
                        else:
                            current_segment.append(item)
                        prev_label = curr_label
                    segments.append(current_segment)
                else:
                    segments = []

                html_content = "<html><body>"
                for segment in segments:
                    segment_label = segment[0]["label"]
                    rgba_color = get_rgba_color(segment_label, 0.6)
                    html_content += f'<div style="background-color:{rgba_color}; padding: 5px; margin:5px 0;">'
                    for item in segment:
                        sentence = item["sentence"]
                        label = item["label"]
                        html_content += (
                            f'<p style="margin:0;">{sentence} '
                            f'<span style="font-size: 0.8em; color: #333;">({label})</span></p>'
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

        # --- Serve index.html ---
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            try:
                with open('index.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    self.wfile.write(html_content.encode('utf-8'))
            except FileNotFoundError:
                error_message = "<html><body><h1>index.html not found</h1></body></html>"
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
