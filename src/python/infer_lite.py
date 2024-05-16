import megenginelite
import cv2
import numpy as np
network = megenginelite.LiteNetwork()
network.load("model/lenet.mge")
input_tensor = network.get_io_tensor("data")
print(input_tensor.layout)

def process(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(32,32))
    image = np.array(255-image)
    return image
image = cv2.imread("image/handwrittern-digit.png")
processed_image = process(image=image)
processed_image = processed_image.reshape(input_tensor.layout.shapes)
input_tensor.set_data_by_copy(processed_image)
# print(f"Tensor is on device {input_tensor.device_type}")
print(processed_image.dtype,processed_image.shape,processed_image.nbytes)
print(input_tensor.nbytes)

network.forward()
network.wait()
output_names = network.get_all_output_name()
for name in output_names:
    output_tensor = network.get_io_tensor(name)
    print(f"name:{name} \noutput_tensor:{output_tensor} \noutput_data:{output_tensor.to_numpy()}")
