import megengine
import megengine.functional
from model import Net
import cv2
import numpy as np
from megengine.jit import trace  # 使用trace装饰器切换到静态图模式
import numpy as np
from megengine import Tensor

check_pointpath = "model/checkpoint.pkl" # python pickle
checkpoint = megengine.load(check_pointpath)
if "state_dict" in checkpoint: # 模型参数的字典
    state_dict = checkpoint["state_dict"]
model = Net()
model.load_state_dict(state_dict=state_dict)

def process(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(32,32))
    image = np.array(255-image)
    return image
image = cv2.imread("image/handwrittern-digit.png")
processed_image = process(image=image)
processed_image = megengine.Tensor(processed_image).reshape(1, 1, 32, 32)
logit = model(processed_image)
label = megengine.functional.argmax(logit).item()
print(label)

@trace(symbolic=True, capture_as_const=True)
def infer_func(data, *, model):
    pred = model(data)
    pred = megengine.functional.argmax(pred)
    return pred

data = np.random.random([1,1,32,32]).astype(np.float32)
res = infer_func(processed_image,model=model)
print(res)
infer_func.dump("model/lenet.mge",arg_names=["data"])
