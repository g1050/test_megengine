import megengine.functional as F
import megengine.module as M
class Net(M.Module):
    def __init__(self):
        super().__init__() # 调用父类初始化
        self.conv1 = M.Conv2d(1,6,5)
        self.conv2 = M.Conv2d(6,16,5)
        self.fc1 = M.Linear(16*5*5,120)
        self.fc2 = M.Linear(120,84)
        self.classifier = M.Linear(84,10)
        self.relu = M.ReLU()
        self.pool = M.MaxPool2d(2,2)
    
    def forward(self, inputs): # 定义forward方法前向计算
        x = self.pool(self.relu(self.conv1(inputs)))
        x = self.pool(self.relu(self.conv2(x)))
        x = F.flatten(x,1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.classifier(x)
        return x