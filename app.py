import streamlit as st
from fastai import *
from fastai.vision import *
import fastai
import PIL 
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
# from PIL import Image
# inferencer = load_learner(path)
def load_model(path=".", model_name="model.pkl"):
    learn = load_learner(path, fname=model_name)
    return learn ``
model = load_model('models')






def predict(img, n: int = 3) -> Dict[str, Union[str, List]]:
    pred_class, pred_idx, outputs = model.predict(img)
    pred_probs = outputs / sum(outputs)
    pred_probs = pred_probs.tolist()
    predictions = []
    for image_class, output, prob in zip(model.data.classes, outputs.tolist(), pred_probs):
        output = round(output, 1)
        prob = round(prob, 2)
        predictions.append(
            {"class": image_class.replace("_", " "), "output": output, "prob": prob}
        )

    predictions = sorted(predictions, key=lambda x: x["output"], reverse=True)
    predictions = predictions[0:n]
    return {"class": str(pred_class), "predictions": predictions}




def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = (512, 512), method = 'bicubic')
    img = img / 255.0
    return img


img_bytes = st.file_uploader("Squash It!!", type=['png', 'jpg', 'jpeg'])
if img_bytes is not None:
    # image = PIL.create(file_uploaded)
    st.write("Image Uploaded Successfully:")
    img = PIL.Image.open(img_bytes)
    img_pil = PIL.Image.open(img_bytes)
    img_tensor = T.ToTensor()(img_pil)
    img_fastai = Image(img_tensor)
    x = st.write(predict(img_fastai,4))