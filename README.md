---
title: CNN
app_file: main.py
sdk: gradio
sdk_version: 5.35.0
license: mit
language:
- en
metrics:
- accuracy
pipeline_tag: image-classification
python_version: 3.12.10
datasets:
- uoft-cs/cifar10
tags:
- image
- classification
- cifar
- cnn
---

# CNN with CIFAR-10

A PyTorch implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset, achieving **81.45% test accuracy**.

## Architecture

![CNN Architecture](assets/architecture.png)

The CNN model consists of
### Convolutional Layers:
- **Conv1**: 3 → 32 channels, 3x3 kernel, padding=1
- **Conv2**: 32 → 64 channels, 3x3 kernel, padding=1  
- **Conv3**: 64 → 128 channels, 3x3 kernel, padding=1

### Others
- **Batch Normalization** after each convolutional layer
- **MaxPooling2D** (2x2) for downsampling
- **ReLU** activation functions
- **Fully Connected Layers**: 2048 → 512 → 10
- **Dropout** (50%) for regularization


## Getting Started

### Prerequisites
- Python 3.12+
- PyTorch 2.7.1+
- torchvision 0.22.1+

> [!TIP]
> This project was developed with `uv`, so it is best to use `uv` for project management.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahuletto/cnn
   cd CNN
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the model

1. Run the main file
```bash
python main.py
```

You can play around in the gradio interface

https://github.com/user-attachments/assets/1f742c32-79bd-4d16-a74f-68c241f4a841

## Model Code
```py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)  # 32x32 -> 16x16
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # 16x16 -> 8x8
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # 8x8 -> 4x4
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```


## Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Epochs**: 50

> Best model checkpoint was saved at epoch 49 with validation loss of 0.6553.


# Model
There are two CNN models in `cnn/` folder
- `model.pt`
- `model-old.pt`

`model.pt` was trained with `BatchNorm2d` to reach 81.45% accuracy in CIFAR-10 dataset
`model-old.pt` was trained without fine tuning which gets 75% accuracy in CIFAR-10 dataset

### Accuracy:

Total Accuracy: `81.45%`

- **Airplane**: `84.60%`
- **Automobile**: `93.20%`
- **Bird**: `76.90%`
- **Cat**: `69.70%`
- **Deer**: `77.20%`
- **Dog**: `64.00%`
- **Frog**: `89.30%`
- **Horse**: `82.10%`
- **Ship**: `89.60%`
- **Truck**: `87.90%`

![Accuracy Benchmark](assets/accuracy.png)

---

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Convolutional Neural Networks for Visual Recognition (CS231n)](http://cs231n.stanford.edu/)
- [Deep Learning Book - Ian Goodfellow](https://www.deeplearningbook.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.