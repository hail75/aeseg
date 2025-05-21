import albumentations as albu
import gradio as gr
import torch
import numpy as np
from PIL import Image
from architecture import MANet
from utils import label2rgb

# Model and weights path'
MODEL_PATH = 'pretrained_weights/fine_tune_manet_model.pth'
N_CLASSES = 6
IMG_SIZE = 256

def load_model():
    model = MANet(num_classes=N_CLASSES)
    state = torch.load(MODEL_PATH, map_location='cpu')
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

model = None

def preprocess(img: Image.Image):
    img = img.convert('RGB')
    if not img.size == (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img)
    img = albu.Normalize()(image=img)['image']
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    return tensor

def predict(image):
    global model
    if model is None:
        model = load_model()
    x = preprocess(image)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    mask_rgb = label2rgb(pred)
    mask_img = Image.fromarray(mask_rgb)
    return image, mask_img

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Input Image'),
    outputs=[gr.Image(type='pil', label='Original'), gr.Image(type='pil', label='Predicted')],
    title='Demo MANet',
    description='Aerial Image Segmentation using MANet',
)

if __name__ == '__main__':
    demo.launch()
