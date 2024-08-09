import torch
import torch.nn as nn
import voyageai
import os
from dotenv import load_dotenv


load_dotenv('keys.env')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api_key_vo=os.getenv('api_key_vo')


def embed_text_vo(text, api_key):
    vo = voyageai.Client(api_key)
    response = vo.embed(
        texts=[text],
        model="voyage-multilingual-2",
        input_type="document"
    )
    return response.embeddings[0]

class MulticlassModel1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassModel1, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.layer2 = nn.Linear(512, 1024)
        self.layer3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

model = MulticlassModel1(input_size=1024, num_classes=2).to(device)

model.load_state_dict(torch.load('sharh_model.pth', weights_only=True))

# Set the model to evaluation mode
model.eval()

label_to_numeric = {'Not Based on Experience': 0,
                    'Based on Experience':1}

def classify_text(input_text, model, label_map, emded, api_key):
    preprocessed_text = input_text
    embedding = emded(preprocessed_text, api_key)
    tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    tensor = tensor.to(device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()

    inverted_label_map = {v: k for k, v in label_map.items()}
    class_probabilities = {inverted_label_map.get(i, "other"): prob for i, prob in enumerate(probabilities[0])}
    sorted_class_probabilities = dict(sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True))

    return sorted_class_probabilities



def find_class(input_text):
  result_label = []
  result_prob = []
  class_probabilities = classify_text(input_text, model, label_to_numeric,embed_text_vo, api_key_vo)
  for i in class_probabilities:
    result_label.append(i)
    result_prob.append(class_probabilities[i])

  probability_percentage = result_prob[0] * 100  # Convert to percentage

  resoponse = {
      'Class': result_label[0],
      'Probability': f'{probability_percentage:.2f}'
  }
  return f"Class: {result_label[0]}", f"Probability: {probability_percentage:.2f}%"








print(find_class('ovqat'))






