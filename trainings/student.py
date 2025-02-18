import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(16 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 141)
        self.fc1 = prune.identity(self.fc1, "weight") 
        self.fc2 = prune.identity(self.fc2, "weight")

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def prune(self, method="l1_unstructured", rate=0.5):
        
        if method == "l1_unstructured":
            prune.l1_unstructured(self.fc1, 'weight', amount=rate)
            prune.l1_unstructured(self.fc2, 'weight', amount=rate)

        elif method == "global":
            parameters_to_prune = [
                (self.fc1, 'weight'),
                (self.fc2, 'weight'),
            ]
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=rate)

        elif method == "random":
            prune.random_unstructured(self.fc1, 'weight', amount=rate)
            prune.random_unstructured(self.fc2, 'weight', amount=rate)

        elif method == "structured":
            prune.ln_structured(self.fc1, 'weight', amount=rate, n=2, dim=0)
            prune.ln_structured(self.fc2, 'weight', amount=rate, n=2, dim=0)

    def remove_pruning(self):
        
        prune.remove(self.fc1, 'weight')
        prune.remove(self.fc2, 'weight')

        