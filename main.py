import os
import gradio as gr
import numpy as np
from torch import device, cuda
from safetensors.torch import load_file
from modules.model import Net
from train import get_transforms

def load_model(model_path):
	global dev
	f = load_file(model_path)
	model = Net().to(dev)
	model.load_state_dict(f)
	return model

def predict(img):
	global model, dev
	if img is None:
		return None
	input = get_transforms()(img.resize((28, 28))).to(dev)
	output = model(input) # returns logits
	prob_sorted, indices = output.sort(descending=True)
	prob_sorted, indices = prob_sorted.cpu().detach().numpy(), indices.cpu().detach().numpy()
	prob_sorted = np.exp(prob_sorted) # reverse the log used in log_softmax()
	return {str(i): float(prob) for i, prob in zip(indices[0], prob_sorted[0])}

def main():
	global model, dev
	dev = device('cuda' if cuda.is_available() else 'cpu')
	model = load_model(os.path.join('models', 'model_1.safetensors'))

	img_size = 512

	with gr.Blocks(title='Handwriting Recognition GUI') as gui:
		gr.Markdown('''# Handwriting Recognition GUI
A simple interface to recognize handwriting using deep learning models''')
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
			
		in_img.change(predict, in_img, label)

	gui.launch(inbrowser=True)

if __name__ == '__main__':
	main()