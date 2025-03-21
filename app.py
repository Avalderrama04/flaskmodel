import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import io

class DenseNet121(nn.Module):
    def __init__(self, num_classes=13):
        super(DenseNet121, self).__init__()
        self.features = models.densenet121(pretrained=True).features
        for param in self.features.parameters():
            param.requires_grad = True
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)    
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

app = Flask(__name__)


MODEL_PATH = "/Users/arthe/umlproject/webapp/DenseNet121_64_lr0.0003_0-5.pt"
model = DenseNet121()  
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

def preprocess_image(image):
    image = Image.open(io.BytesIO(image))
    image = transform(image).unsqueeze(0) 
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = preprocess_image(file.read())

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        prediction = probabilities.argmax().item()

    return jsonify({'prediction': prediction, 'confidence': probabilities[0][prediction].item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
