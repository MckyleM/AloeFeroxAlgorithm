import torch.nn as nn
class SingleLabelCNN(nn.Module):
    def __init__(self, num_classes):
        super(SingleLabelCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        return nn.functional.log_softmax(x, dim=1)