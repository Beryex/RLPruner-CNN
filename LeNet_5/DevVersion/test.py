import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from thop import profile
from utils import get_CIFAR10_test_dataloader, get_MNIST_test_dataloader


# move the LeNet Module into the corresponding device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_img(data, save_path):
    for i in tqdm(range(len(data))):
        img, label = data[i]
        img.save(os.path.join(save_path, str(i) + '-label-' + str(label) + '.png'))

def visualize_layer():
    images = ['./DataImages-Test/62-label-9.png']
    for i in range(10):
    # process image
        image = Image.open(images[i])

        transforms_i = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor()])

        image = transforms_i(image)
        print(f"Image shape before: {image.shape}")
        image = image.unsqueeze(0)
        print(f"Image shape after: {image.shape}")

        image = image.to(device)

        # process model
        model = torch.load('models/mnist_1708800786.pkl')
        layers = []
        layers.append(model.conv1)
        layers.append(model.conv1_activation_func)
        layers.append(model.pool1)
        layers.append(model.conv2)
        layers.append(model.conv2_activation_func)
        layers.append(model.pool2)
        layers.append(model.fc1)
        layers.append(model.fc1_activation_func)
        layers.append(model.fc2)
        layers.append(model.fc2_activation_func)
        layers.append(model.fc3)
        layers.append(model.fc3_activation_func)

        # get processed image through model
        outputs = []
        names = []
        layer_count = 0
        outputs.append(image)
        names.append('Input Figure')

        model.eval()
        for layer in layers[0:]:
            layer_count += 1
            if layer_count == 7:
                image = image.view(image.shape[0], -1) 
            image = layer(image)
            if layer_count < 7:
                outputs.append(image)
                names.append(str(layer))
        
        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]  
            processed.append(gray_scale.data.cpu().numpy())
        
        fig = plt.figure(figsize=(30, 50))
        for i in range(len(processed)):  
            a = fig.add_subplot(5, 4, i+1)
            img_plot = plt.imshow(processed[i])
            a.axis("off")
            a.set_title(names[i].split('(')[0], fontsize=30)

        plt.savefig('feature_maps.png', bbox_inches='tight')


def test():
    mnist_test_loader = get_MNIST_test_dataloader(
        num_workers=4,
        batch_size=1024,
        shuffle=True
    )

    original_para_num = 0.0
    original_FLOPs_num = 0.0
    compressed_para_num = 0.0
    compressed_FLOPs_num = 0.0
    input = torch.rand(256, 1, 32, 32).to(device)

    # initialize the testing parameters
    top1_correct_num = 0.0
    top3_correct_num = 0.0

    # begin testing
    model = torch.load('models/LeNet_Original_1710607419.pkl')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (test_x, test_label) in enumerate(mnist_test_loader):
            # move test data to device
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            # get predict y and predict its class
            outputs = model(test_x)
            _, preds = outputs.topk(3, 1, largest=True, sorted=True)
            top1_correct_num += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
            top3_correct = test_label.view(-1, 1).expand_as(preds) == preds
            top3_correct_num += top3_correct.any(dim=1).sum().item()
    # calculate the accuracy and print it
    top1_accuracy = top1_correct_num / len(mnist_test_loader.dataset)
    top3_accuracy = top3_correct_num / len(mnist_test_loader.dataset)
    print('Original model has top1 accuracy: %f, top3 accuracy: %f' %(top1_accuracy, top3_accuracy))
    original_FLOPs_num, original_para_num = profile(model, inputs = (input, ), verbose=False)
        
    
    # initialize the testing parameters
    top1_correct_num = 0.0
    top3_correct_num = 0.0

    # begin testing
    model = torch.load('models/LeNet_Compressed_1710861836.pkl')
    model = model.to(device)
    print(model)
    model.eval()
    with torch.no_grad():
        for idx, (test_x, test_label) in enumerate(mnist_test_loader):
            # move test data to device
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            # get predict y and predict its class
            outputs = model(test_x)
            _, preds = outputs.topk(3, 1, largest=True, sorted=True)
            top1_correct_num += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
            top3_correct = test_label.view(-1, 1).expand_as(preds) == preds
            top3_correct_num += top3_correct.any(dim=1).sum().item()
    # calculate the accuracy and print it
    top1_accuracy = top1_correct_num / len(mnist_test_loader.dataset)
    top3_accuracy = top3_correct_num / len(mnist_test_loader.dataset)
    print('Compressed Model has top1 accuracy: %f, top3 accuracy: %f' %(top1_accuracy, top3_accuracy))
    compressed_FLOPs_num, compressed_para_num = profile(model, inputs = (input, ), verbose=False)
    
    # get compressed ratio
    FLOPs_compressed_ratio = compressed_FLOPs_num / original_FLOPs_num
    Para_compressed_ratio = compressed_para_num / original_para_num
    print('Original FLOPs: %f, Parameter Num: %f' %(original_FLOPs_num, original_para_num))
    print('Compressed FLOPs: %f, Parameter Num: %f' %(compressed_FLOPs_num, compressed_para_num))
    print('We achieve FLOPS compressed ratio: %f, parameter number compressed ratio: %f' %(FLOPs_compressed_ratio, Para_compressed_ratio))

if __name__ == '__main__':
    '''test_data = datasets.MNIST(root="./uncompressed_data/", train=False, download=True)
    saveDirTest = './DataImages-Test'
    if not os.path.exists(saveDirTest):
        os.mkdir(saveDirTest)
    save_img(test_data, saveDirTest)'''
    test()
    # visualize_layer()