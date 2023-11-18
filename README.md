# Hand-Written Recognition GUI

A simple GUI to train and test models for hand-written digits and characters recognition.
![GUI screenshot](assets/gui_screenshot.png)

## Installation
Clone the repo and install necessary dependencies (tested in Python 3.9):
```
git clone https://github.com/Pikurrot/hand-write-gui.git
pip install -r requirements.txt
```

## Usage
### Train a model
- Run `python3 train.py` if you are in **Linux / MacOS** or `python train.py` if in Windows.
- The training data will be downloaded to `data` and a model will be trained. This process may take a while.
- After the training has finished, the model will be saved in `models` as `model_0.safetensors`.

### Run the GUI
- Run the `hand-write-gui.sh` file if you are in **Linux / MacOS**, or `hand-write-gui.bat` if in **Windows**.  
*Ensure you have python in your environment variables.*
- A GUI will open in a new tab of your browser, start drawing in the canvas.
