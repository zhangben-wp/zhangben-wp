import torch
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from convnet import ConvNet


batch_size = 16
# 准备数据集

data_dir = {'train': 'MSTAR-SOC\\train', 'test': 'MSTAR-SOC\\test'}

# 定义训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloader加载数据
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_transform, test_transform = data_transforms['train'], data_transforms['test']


trainset = datasets.ImageFolder(data_dir['train'], transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


testset = datasets.ImageFolder(data_dir['test'], transform=train_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


classes_name = trainset.classes



# 调用网络
model = ConvNet(num_classes=10)
model.to(device)

# 定义损失函数



# 优化器
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 训练
n_epochs = 30
total_train_step = 0
total_test_step = 0

# writer = SummaryWriter('log')

for epoch in range(n_epochs):
    # 训练
    total_train_accuracy = 0
    total_train_loss = 0
    total_train_sample = 0
    model.train()
    for batch_idx, data in enumerate(trainloader):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        # 模型训练
        
        acc_train = (outputs.argmax(1) == targets).sum()
        total_train_accuracy += acc_train.item()
        total_train_loss += loss.item()
        total_train_sample += imgs.shape[0]



        # total_train_step += 1
        # if total_train_step % 100 == 0:
        #     print('训练次数：{}，Loss：{}'.format(total_train_step, loss.item()))

    train_loss = total_train_loss / (batch_idx + 1)
    train_acc = total_train_accuracy / total_train_sample

    # 测试
    total_test_accuracy = 0
    total_test_loss = 0
    total_test_sample = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            acc_test = (outputs.argmax(1) == targets).sum()
            total_test_accuracy += acc_test.item()
            total_test_loss += loss.item()
            total_test_sample += imgs.shape[0]

        test_loss = total_test_loss / (batch_idx + 1)
        test_acc = total_test_accuracy / total_test_sample

    print('Epoch: {}/{} - loss: {:.4f} - acc: {:.4f} - valt_loss: {:.4f} - val_acc: {:.4f}'.format(
        epoch + 1, n_epochs,
        train_loss, train_acc,
        test_loss, test_acc
    ))
torch.save(model.state_dict(), 'weights.pth')
    # writer.add_scalars('log/loss', {'train loss': train_loss, 'test loss': test_loss}, epoch)
    # writer.add_scalars('log/acc', {'train acc': train_acc, 'test acc': test_acc}, epoch)

    # if (epoch + 1) % 10 == 0:
    #     torch.save(model, 'convnet_{}.pth'.format(epoch))
    #     print('model has been saved')

# writer.close()
