import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torchvision import transforms
from cnn import CNN

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

model = CNN()
model.load_state_dict(torch.load("cnn/model.pt", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def predict(image):
    if image is None:
        return {}

    image = Image.fromarray(image).convert("RGB")
    image_tensor = transform(image)
    image_tensor = (image_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    return {classes[i]: float(probabilities[i]) for i in range(len(classes))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=10),
    title="CNN Classifier",
    description="Upload an image to classify it into one of 10 CIFAR-10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck",
    examples=[
        ["examples/1.png"],
        ["examples/2.png"],
        ["examples/3.png"],
        ["examples/4.png"],
        ["examples/5.png"],
        ["examples/6.png"],
        ["examples/7.png"],
    ],
)

if __name__ == "__main__":
    demo.launch(share=True, pwa=True)
