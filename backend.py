from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms, models
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Shared image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Load fruit classifier model ---
fruit_model = models.resnet50(pretrained=False)
fruit_model.fc = torch.nn.Linear(fruit_model.fc.in_features, 4)  # 4 fruit classes
fruit_model.load_state_dict(torch.load("model_output_1/resnet50_fruit_classifier.pth", map_location="cpu"))
fruit_model.eval()

fruit_classes = ["Avocado", "Banana", "Carrot", "Pineapple"]

# --- Load variation classifier model ---
variation_model = models.resnet50(pretrained=False)
variation_model.fc = torch.nn.Linear(variation_model.fc.in_features, 12)  # 3 variation classes
variation_model.load_state_dict(torch.load("model_output_2/resnet50_fruit_classifier.pth", map_location="cpu"))
variation_model.eval()

variation_classes = [
    "Baby_Carrot", "Chunks","Halved_With_Pit",
    "In_A_Bunch","Mashed_Diced","Peeled_Chopped",
     "Peeled_Half-Peeled", "Rings", "Whole",
     "Whole_Avacado", "Whole_Unpealed_Banana", "Whole_Unpeeled_Carrot"

]

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # --- Predict fruit type ---
    with torch.no_grad():
        fruit_output = fruit_model(image)
        _, fruit_pred = torch.max(fruit_output, 1)
        fruit_label = fruit_classes[fruit_pred.item()]

        # --- Predict variation ---
        variation_output = variation_model(image)
        _, variation_pred = torch.max(variation_output, 1)
        variation_label = variation_classes[variation_pred.item()]

    return jsonify({
        "fruit": fruit_label,
        "variation": variation_label
    })

if __name__ == "__main__":
    app.run(debug=True)
