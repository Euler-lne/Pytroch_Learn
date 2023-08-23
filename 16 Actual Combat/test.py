from PIL import Image
import torchvision
from model import Euler
import torch

answer = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

img_path = "16 Actual Combat/imgs/6.png"
image = Image.open(img_path)

image = image.convert('RGB')
# 因为png格式是四个通道，处理RGB三个通道之外还有一个透明通道。所以只保留起RGB通道。

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

image = transform(image)

euler = Euler()
# euler.load_state_dict(torch.load("euler.pth",map_location=torch.device('cpu')))
# 上面的代码是用于用GPU训练但是电脑上没有GPU的情况
euler.load_state_dict(torch.load("euler.pth"))
image = torch.reshape(image, (1, 3, 32, 32))
# 原因输入图片是4维，有一维是batch_size，原因看7 nn.Module

# 注意要加入下面两行
euler.eval()
with torch.no_grad():
    output = euler(image)

print(answer[output.argmax(1)])
