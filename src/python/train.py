from os.path import expanduser
from megengine.data.dataset import MNIST
import megengine.data as data
import megengine.data.transform as T
import megengine.functional as F
from model import Net
import megengine.optimizer as optim
import megengine.autodiff as autodiff
import megengine
import os
import cv2
import numpy as np

def get_dataset(dataset_path):
    MNIST_DATA_PATH=expanduser(dataset_path)
    print(MNIST_DATA_PATH)
    train_dataset = MNIST(MNIST_DATA_PATH, train=True,download=False)
    test_dataset = MNIST(MNIST_DATA_PATH, train=False,download=False)

    train_sampler =  data.RandomSampler(train_dataset,batch_size=64) # 不放回（不重复选取）地随机采样。
    test_sampler = data.RandomSampler(test_dataset,batch_size=64)
    transform = T.Compose([
        T.Normalize(0.1307*255, 0.3081*255),
        T.Pad(2),
        T.ToMode("CHW"),
    ])

    train_dataloader = data.DataLoader(train_dataset,train_sampler,transform=transform)
    test_dataloader = data.DataLoader(test_dataset,test_sampler,transform=transform)
    return train_dataloader,test_dataloader,train_dataset,test_dataset

def train_lenet(train_dataloader,test_dataloader,train_dataset,test_dataset):
    model = Net()
    gm = autodiff.GradManager().attach(model.parameters()) # 对梯度进行管理 
    optimizer = optim.SGD( # 反向传播，参数更新，选用SGD
        model.parameters(),
        lr=0.01,momentum=0.9,weight_decay=5e-4
    )

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
    return model

def process(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(32,32))
    image = np.array(255-image)
    return image

def main():
    train_dataloader,test_dataloader,train_dataset,test_dataset = get_dataset("../dataset/MINIST")
    model = train_lenet(train_dataloader,test_dataloader,train_dataset,test_dataset)
    
    # 单张图片推理验证
    image = cv2.imread("image/handwrittern-digit.png")
    processed_image = process(image=image)
    logit = model(megengine.Tensor(processed_image).reshape(1, 1, 32, 32))
    pred = F.argmax(logit, axis=1).item()
    print(f"pred: {pred}")

if __name__ == "__main__":
    main()
