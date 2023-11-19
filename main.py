import os
import gradio as gr
import numpy as np
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from modules.model import Net
from modules.utils import list_models
from train import get_transforms

def load_model_local(model_path):
	global model, device
	f = load_file(model_path)
	model = Net().to(device)
	model.load_state_dict(f)

def load_model_huggingface(model_path):
	global model, device
	repo_id, filename = model_path.rsplit('/', 1)
	# check if model is already downloaded
	if not os.path.exists(os.path.join('models', filename)):
		hf_hub_download(repo_id=repo_id,
					filename=filename,
					local_dir='models',
					local_dir_use_symlinks=False)
	load_model_local(os.path.join('models', filename))

def update_model(model_path_saved, model_path_hf, model_path_upload, source, logs):
	global model, device
	if source == 'Saved':
		if not model_path_saved:
			return logs + 'Could not load model: No model selected\n'
		try:
			load_model_local(os.path.join('models', model_path_saved))
		except Exception as e:
			return logs + f'Could not load model: {str(e)}\n'
	elif source == 'Huggingface':
		if not model_path_hf:
			return logs + 'Could not load model: No model path to Huggingface providedv'
		try:
			load_model_huggingface(model_path_hf)
		except Exception as e:
			return logs + f'Could not load model: {str(e)}\n'
	elif source == 'Uploaded':
		if not model_path_upload:
			return logs + 'Could not load model: No model uploaded\n'
		try:
			load_model_local(model_path_upload.name)
		except Exception as e:
			return logs + f'Could not load model: {str(e)}\n'
			
	return logs + f'Model updated successfully. Moved to {device}\n'

def predict(img, logs):
	global model, device
	if img is None:
		return None, logs
	if model is None:
		gr.Info('No model loaded. Using default from Huggingface...')
		logs += 'No model loaded. Using default from Huggingface\n'
		try:
			load_model_huggingface('Pikurrot/digitnet-tiny/digitnet-tiny.safetensors')
		except Exception as e:
			gr.Error(f'Could not load model: {str(e)}')
			logs += f'Could not load model: {str(e)}\n'
			return None, logs
	try:
		input = get_transforms()(img.resize((28, 28))).to(device)
		output = model(input) # returns logits
		prob_sorted, indices = output.sort(descending=True)
		prob_sorted, indices = prob_sorted.cpu().detach().numpy(), indices.cpu().detach().numpy()
		prob_sorted = np.exp(prob_sorted) # reverse the log used in log_softmax()
		return {str(i): float(prob) for i, prob in zip(indices[0], prob_sorted[0])}, logs
	except Exception as e:
		gr.Error(f'Model prediction failed: {str(e)}')
		logs += f'Model prediction failed: {str(e)}\n'
		return None, logs

def main():
	global model, device
	model = None
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	file_types = ['safetensors']
	models = list_models(file_types)

	img_size = 512

	with gr.Blocks(title='Handwriting Recognition GUI') as gui:
		gr.Markdown('''# Handwriting Recognition GUI
A simple interface to recognize handwriting using deep learning models''')
		with gr.Tab('Test model'):
			with gr.Row():
				in_img = gr.Image(
					height=img_size,
					width=img_size,
					min_width=img_size,
					shape=(256, 256),
					image_mode='L',
					type='pil',
					source='canvas',
					brush_radius=20,
					invert_colors=True,
					show_label=False
				)
				label = gr.Label(show_label=False)
		with gr.Tab('Load model'):
			with gr.Row():
				with gr.Column():
					model_path_saved = gr.Dropdown(models, value=None, label='Saved models')
					model_path_hf = gr.Textbox('Pikurrot/digitnet-tiny/digitnet-tiny.safetensors', placeholder='Pikurrot/digitnet-tiny/digitnet-tiny.safetensors', label='Download from Huggingface')
					model_path_upload = gr.File(value = None, label='Upload model', file_types=file_types)
					source = gr.Radio(['Saved', 'Huggingface', 'Uploaded'], file_types=file_types, value='Huggingface', label='Choose model source')
					update_btn = gr.Button(value='Update')
				logs = gr.Textbox(None, placeholder='No model loaded', lines=25, max_lines=25, label='Logs', interactive=False, autoscroll=True)

		in_img.change(predict, [in_img, logs], [label, logs])
		update_btn.click(
			update_model,
			[model_path_saved, model_path_hf, model_path_upload, source, logs],
			logs)

	gui.queue().launch(inbrowser=True)

if __name__ == '__main__':
	main()
