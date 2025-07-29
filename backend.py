from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms, models
import io
from flask_cors import CORS


app = Flask(__name__)
CORS(app)



# --- Load model architecture first ---
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 = number of fruit classes

# --- Load weights into the model ---
model.load_state_dict(torch.load("model_output_1/resnet50_fruit_classifier.pth", map_location="cpu"))
model.eval()

# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Labels used during training ---
class_names = ["Avocado", "Banana", "Carrot", "Pineapple"]  # Update if needed

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    
    
    
    image = Image.open(file.stream).convert("RGB")
    
    
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
