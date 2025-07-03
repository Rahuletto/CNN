# CNN with CIFAR-10

A PyTorch implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset, achieving **81.45% test accuracy**.

## Architecture

![CNN Architecture](benchmarks/architecture.png)

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
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Epochs**: 50

> Best model checkpoint was saved at epoch 49 with validation loss of 0.6553.


## Performance

![Training Loss](benchmarks/loss.png)
Reaching 0.7227 in Train loss and 0.6557 in Validation loss at epoch 50


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

![Accuracy Benchmark](benchmarks/accuracy.png)

---

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Convolutional Neural Networks for Visual Recognition (CS231n)](http://cs231n.stanford.edu/)
- [Deep Learning Book - Ian Goodfellow](https://www.deeplearningbook.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.