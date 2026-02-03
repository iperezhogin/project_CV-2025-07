import warnings
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel
from torchvision.transforms import v2
from PIL import Image
# import io
# from pathlib import Path

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"


class BackboneModel():
    def __init__(self, task='feature_extraction'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.task = task
        if task == 'classification':
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base-imagenet1k-1-layer')
            self.model = AutoModelForImageClassification.from_pretrained('facebook/dinov2-base-imagenet1k-1-layer').to(device)
            self.transforms = v2.Compose([
                v2.ToTensor(),
                v2.Resize((224, 224)),
            ])
        else:
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.model = AutoModel.from_pretrained('facebook/dinov2-base')

    def preprocess(self, img):
        # img = Image.open(io.BytesIO(img)).convert('RGB')
        image = Image.open(img).convert('RGB')
        image: torch.Tensor = self.transforms(image).unsqueeze(0)
        return image

    def predict_logits(self, img):
        assert self.task == 'classification', f"Incorrect task name ({self.task} instead of 'classification')"
        img = self.preprocess(img)
        inputs = self.processor(images=img.float(), return_tensors="pt", do_resize=False, do_rescale=False)
        outputs = self.model(**inputs.to(self.device))
        emb = outputs.logits.detach().flatten().cpu().numpy()
        return emb

    def extract_embedding(self, img):
        image = Image.open(img).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.detach()[:, 0, :].flatten().cpu().numpy()
