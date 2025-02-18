import torchvision.models as models
import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(self, num_classes=141, feature_extract=True):
        super(TeacherModel, self).__init__()
        self.model = models.inception_v3(weights='DEFAULT')
        
        if feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
        