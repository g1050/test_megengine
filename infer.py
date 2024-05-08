# import megengine as mge
# from model import LeNet

# checkpoint_path = "output/checkpoint.pkl"
# checkpoint = mge.load(checkpoint_path)
# if "state_dict" in checkpoint:
#     state_dict = checkpoint["state_dict"]

# model = LeNet()
# model.load_state_dict(state_dict)

# # Input data (request from the client)
# # ...

# pred = model(data)

# # Return the result back to the client

import megengine
import megengine.functional
from model import Net
check_pointpath = "model/checkpoint.pkl" # python pickle
checkpoint = megengine.load(check_pointpath)
if "state_dict" in checkpoint: # 模型参数的字典
    state_dict = checkpoint["state_dict"]
model = Net()
model.load_state_dict(state_dict=state_dict)

import cv2
import numpy as np

def process(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(32,32))
    image = np.array(255-image)
    return image
image = cv2.imread("image/handwrittern-digit.png")
processed_image = process(image=image)
processed_image = megengine.Tensor(processed_image).reshape(1, 1, 32, 32)
# def infer_func(processed_img):
#     logit = model(processed_img)
#     label = F.argmax(logit).item()
#     return label
logit = model(processed_image)
label = megengine.functional.argmax(logit).item()
print(label)

from megengine.jit import trace  # 使用trace装饰器切换到静态图模式
@trace(symbolic=True, capture_as_const=True)
def infer_func(data, *, model):
    pred = model(data)
    pred = megengine.functional.argmax(pred)
    return pred

# 执行一次上述函数得到动态图信息
# import numpy as np
# from megengine import Tensor

# data = np.random.random([1, 1, 32, 32]).astype(np.float32)
# infer_func(Tensor(data), model=model)
import numpy as np
from megengine import Tensor
data = np.random.random([1,1,32,32]).astype(np.float32)
res = infer_func(processed_image,model=model)
print(res)
# infer_func.dump("snetv2_x100_deploy.mge", arg_names=["data"]) 
infer_func.dump("lenet.mge",arg_names=["data"])
