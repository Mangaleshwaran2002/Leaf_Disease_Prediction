import gdown
import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

BASE_DIR=os.getcwd()
#url to download weights

if os.path.isdir(f"{BASE_DIR}/weights_yolov8"):
  print("weight's folder already downloaded")
else:
  url = 'https://drive.google.com/drive/folders/1-V2MdBzL7M7gTqxzyPeK-cxlBm-5JwUJ?usp=sharing'
  gdown.download_folder(url)
if os.path.isdir(f"{BASE_DIR}/images"):
  print("image's folder already downloaded")
else:
  url = 'https://drive.google.com/drive/folders/1UZJ98GXCx7CyY0gMoxYgNFrk17r1FShL?usp=sharing'
  gdown.download_folder(url)



# Load a pretrained YOLOv8 model
model_path=f"{BASE_DIR}/weights_yolov8/best.pt"
if os.path.isfile(model_path):
  model = YOLO(model_path)

  def predict(img):
    # print(img)
    numpy_array = np.array(img) 
    numpy_array = numpy_array.astype(np.uint8)
    image = Image.fromarray(numpy_array)
    result = model.predict(image,imgsz=320, conf=0.5)
    for i, r in enumerate(result):
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        return im_rgb

  #Create a gradio block
  with gr.Blocks(theme=gr.themes.Soft()) as MyBlock:
      gr.Markdown("Start typing below and then click **Run** to see the output.")
      with gr.Row():
          inp = gr.Image(label="upload a leaf image")
          out = gr.Image(label="predicted leaf image")
      
      examples = gr.Examples(
          examples=[
              f"{BASE_DIR}/images/20211109_122317.jpg",
              f"{BASE_DIR}/images/20211109_122322.jpg",
              f"{BASE_DIR}/images/IMG_20211106_120807.jpg",
              f"{BASE_DIR}/images/IMG_20211106_120833.jpg",
          ],
          inputs=inp
      )
      btn = gr.Button("Run")
      btn.click(fn=predict, inputs=inp, outputs=out)

MyBlock.launch(share=True,debug=True)
