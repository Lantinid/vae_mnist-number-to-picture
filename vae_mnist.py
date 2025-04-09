import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import numpy as np

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("当前设备为cuda")
else:
    print("当前设备为cpu")

# 定义变分自动编码器模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 编码器部分
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc21 = nn.Linear(256, 20)
        self.fc22 = nn.Linear(256, 20)

        # 解码器部分
        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 64 * 7 * 7)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        # 输入数据经过卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)
        # 输入数据经过第一个全连接层
        h1 = torch.relu(self.fc1(x))
        # 计算均值和对数方差
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # 计算标准差
        std = torch.exp(0.5 * logvar)
        # 生成随机噪声
        eps = torch.randn_like(std)
        # 重参数化技巧
        return mu + eps * std

    def decode(self, z):
        # 潜在变量经过全连接层
        h3 = torch.relu(self.fc3(z))
        h4 = torch.relu(self.fc4(h3))
        h4 = h4.view(-1, 64, 7, 7)
        # 经过反卷积层
        h5 = F.relu(self.conv4(h4))
        h6 = F.relu(self.conv5(h5))
        # 输出重构后的图像
        return torch.sigmoid(self.conv6(h6))

    def forward(self, x):
        # 编码输入数据
        mu, logvar = self.encode(x)
        # 重参数化
        z = self.reparameterize(mu, logvar)
        # 解码潜在变量
        return self.decode(z), mu, logvar

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    # 重构损失
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    # KL 散度损失
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # 调整权重
    beta = 0.5
    return BCE + beta * KLD

# 训练模型
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True,transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 定义一个字典来存储数字和潜在变量的映射关系
digit_latent_mapping = {}

# 修改训练函数，训练数字和潜在变量的映射关系
def train(model, train_loader, optimizer, epoch):
    """
    训练VAE模型的一个epoch
    
    参数:
        model: VAE模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前epoch数
    
    返回:
        train_loss: 当前epoch的总损失
    """
    # 设置模型为训练模式
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]	Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        # 训练数字和潜在变量的映射关系
        # 获取当前数字标签
        # 获取对应的潜在变量(均值)并增加批次维度
        # 如果该数字尚未在映射字典中，初始化空列表
        # 将潜在变量添加到对应数字的列表中
        for i in range(len(labels)):
            digit = labels[i].item()
            latent_variable = mu[i].unsqueeze(0)
            if digit not in digit_latent_mapping:
                digit_latent_mapping[digit] = []
            digit_latent_mapping[digit].append(latent_variable)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    return train_loss

def start_train(model):
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 训练模型
    epochs = 30
    # 用于记录每个epoch的损失
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch)
        epoch_losses.append(train_loss)
    # 保存模型
    torch.save(model.state_dict(), 'vae_mnist.pth')
    # 绘制损失随epoch变化的曲线
    plt.plot(range(1, epochs + 1), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.savefig('loss_curve.png')
    plt.show()

# 保存映射关系
def save_mapping():
    torch.save(digit_latent_mapping, 'digit_latent_mapping.pth')

# 加载映射关系
def load_mapping():
    global digit_latent_mapping
    if os.path.exists('digit_latent_mapping.pth'):
        digit_latent_mapping = torch.load('digit_latent_mapping.pth')

# 修改生成图片的函数，根据输入数字获取对应潜在变量
def generate_image(model, digit):
    # 确保模型在评估模式下运行，不进行梯度计算
    model.eval()
    with torch.no_grad():
        if digit in digit_latent_mapping:
            # 随机选择一个潜在变量
            random_index = np.random.randint(0, len(digit_latent_mapping[digit]))
            latent_variable = digit_latent_mapping[digit][random_index].cpu().unsqueeze(0)
            # 将潜在变量移动到与模型相同的设备上
            latent_variable = latent_variable.to(device)
            # 解码潜在变量生成图片
            sample = model.decode(latent_variable).cpu()
            # 调整图片形状
            sample = sample.view(28, 28).numpy()
            # 显示图片
            plt.imshow(sample, cmap='gray')
            plt.show()
        else:
            print('未找到该数字对应的潜在变量，请先训练模型。')

# 用户输入数字生成图片
def generate_from_input(model):
    while True:
        try:
            digit = int(input("请输入一个0 - 9的数字作为要生成的数字(输入-1退出): "))
            if digit == -1:
                break
            if isinstance(digit, int) and 0 <= digit <= 9:
                generate_image(model, digit)
            else:
                print("请输入0 - 9的有效整数!")
        #except ValueError:
        #    print("请输入有效的整数!")
        #捕获其他异常
        except Exception as e:
            print(f"发生错误: {e}")

# 修改主函数，添加保存和加载映射关系的逻辑
def main():
    model = VAE().to(device)
    # 检查模型是否存在
    if os.path.exists('vae_mnist.pth'):
        # 加载模型
        print("模型存在，开始加载模型...")
        model.load_state_dict(torch.load('vae_mnist.pth'))
        # 加载映射关系
        load_mapping()
        # 询问是否继续训练
        choice = input("模型已存在，是否继续训练？(y/n): ")
        if choice.lower() == 'y':
            start_train(model)
            # 保存映射关系
            save_mapping()
    else:
        print("模型不存在，开始训练新模型...")
        start_train(model)
        # 保存映射关系
        save_mapping()
    # 根据用户输入生成图片
    generate_from_input(model)

if __name__ == '__main__':
    main()