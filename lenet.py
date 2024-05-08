from os.path import expanduser
from megengine.data.dataset import MNIST
MNIST_DATA_PATH=expanduser("./dataset/MINIST")
print(MNIST_DATA_PATH)
train_dataset = MNIST(MNIST_DATA_PATH, train=True,download=False)
test_dataset = MNIST(MNIST_DATA_PATH, train=False,download=False)

import megengine.data as data
import megengine.data.transform as T

train_sampler =  data.RandomSampler(train_dataset,batch_size=64) # 不放回（不重复选取）地随机采样。
test_sampler = data.RandomSampler(test_dataset,batch_size=64)
transform = T.Compose([
    T.Normalize(0.1307*255, 0.3081*255),
    T.Pad(2),
    T.ToMode("CHW"),
])

train_dataloader = data.DataLoader(train_dataset,train_sampler,transform=transform)
test_dataloader = data.DataLoader(test_dataset,test_sampler,transform=transform)


# import megengine.functional as F
# import megengine.module as M

# class LeNet(M.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = M.Conv2d(1, 6, 5)
#         self.conv2 = M.Conv2d(6, 16, 5)
#         self.fc1 = M.Linear(16 * 5 * 5, 120)
#         self.fc2 = M.Linear(120, 84)
#         self.classifier = M.Linear(84, 10)

#         self.relu = M.ReLU()
#         self.pool = M.MaxPool2d(2, 2)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = F.flatten(x, 1)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.classifier(x)
#         return x


# model = LeNet()

from model import Net
model = Net()

# import megengine.optimizer as optim
# import megengine.autodiff as autodiff 

# gm = autodiff.GradManager().attach(model.parameters())
# optimizer = optim.SGD(
#     model.parameters(),
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=5e-4
# )

import megengine.optimizer as optim
import megengine.autodiff as autodiff

gm = autodiff.GradManager().attach(model.parameters()) # 对梯度进行管理 
optimizer = optim.SGD( # 反向传播，参数更新，选用SGD
    model.parameters(),
    lr=0.01,momentum=0.9,weight_decay=5e-4
)



# import megengine
# epochs = 10
# model.train()
# for epoch in range(epochs):
#     total_loss = 0
#     for batch_data, batch_label in train_dataloader:
#         batch_data = megengine.Tensor(batch_data)
#         batch_label = megengine.Tensor(batch_label)

#         with gm:
#             logits = model(batch_data)
#             loss = F.nn.cross_entropy(logits, batch_label)
#             gm.backward(loss)
#             optimizer.step().clear_grad()

#         total_loss += loss.item()

#     print(f"Epoch: {epoch}, loss: {total_loss/len(train_dataset)}")

import megengine
import os
epochs = 10
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_data,batch_label in train_dataloader:
        batch_data = megengine.Tensor(batch_data) # 将数据转换为megengine的Tensor格式
        batch_label = megengine.Tensor(batch_label)
        with gm:
            logits = model(batch_data)
            loss = F.nn.cross_entropy(logits,batch_label)
            gm.backward(loss) # 反向传播计算梯度
            optimizer.step().clear_grad()

        total_loss += loss.item()
        print(f"Epoch: {epoch},loss {total_loss/len(train_dataset)}")
    if (epoch + 1) % 5 == 0:
        megengine.save({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
        },
        os.path.join("model", "checkpoint.pkl"))

# model.eval()
# correct, total = 0, 0
# for batch_data, batch_label in test_dataloader:
#     batch_data = megengine.Tensor(batch_data)
#     batch_label = megengine.Tensor(batch_label)

#     logits = model(batch_data)
#     pred = F.argmax(logits, axis=1)
#     correct += (pred == batch_label).sum().item()
#     total += len(pred)

# print(f"Correct: {correct}, total: {total}, accuracy: {float(correct)/total}")

model.eval()
correct,total = 0,0
for batch_data,batch_label in test_dataloader:
    batch_data = megengine.Tensor(batch_data)
    batch_label = megengine.Tensor(batch_label)
    logits = model(batch_data)
    pred = F.argmax(logits,axis=1)
    correct += (pred == batch_label).sum().item()
    total += len(pred)
print(f"Correct {correct},total {total}, accuracy :{float(correct/total)}")

# import cv2
# import numpy as np

# def process(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.resize(image, (32, 32))
#     image = np.array(255 - image)
#     return image

# image = cv2.imread("/data/handwrittern-digit.png")
# processed_image = process(image)

# 单张图片推理验证
# import cv2
# import numpy as np

# def process(image):
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image = cv2.resize(image,(32,32))
#     image = np.array(255-image)
#     return image
# image = cv2.imread("image_path")
# processed_image = process(image=image)