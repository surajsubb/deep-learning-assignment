import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import time
import Model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


def load_image(path):
    img=Image.open(path)
    return img

def enhance(path):
    Original_image_path=path
    with torch.no_grad():
      data_lowlight_og = Image.open(Original_image_path)
      data_lowlight = (np.asarray(data_lowlight_og)/255.0)
      data_lowlight = torch.from_numpy(data_lowlight).float()
      data_lowlight = data_lowlight.permute(2,0,1)
      data_lowlight = data_lowlight.unsqueeze(0)
      model = Model.EnhanceNet()
      model.load_state_dict(torch.load('./snapshots/Epoch96.pth',map_location=torch.device("cpu")))
      start = time.time()
      _,enhanced_image,_ = model(data_lowlight)
    return enhanced_image

st.markdown("# LOL FACE - Enhancement")
st.write("### Low Light Face detection Enhancement Network Inspired from U-Net")
st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if image_file is not None:
    file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
    #st.write(file_details)
    st.write("## Original Image")
    st.image(load_image(image_file))
    
    img=load_image(image_file)
    with open("./temp/temp.png","wb") as f:
        f.write(image_file.getbuffer())
    enhanced_img=enhance("./temp/temp.png")
    st.write("## Successfully Enhanced")
    st.write("## Enhanced Image !")
    torchvision.utils.save_image(enhanced_img,"./temp/enhanced.png")
    enhanced=load_image("./temp/enhanced.png")
    st.image(enhanced)
	    