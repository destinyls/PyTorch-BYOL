import os
import sys
import yaml
import numpy as np
sys.path.append('../')
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer

print(torch.__version__)
torch.manual_seed(0)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

class Classifier(nn.Module):
    '''
    Generalized structure for classifier.
    main parts:
    - backbone
    - mlp heads
    '''
    def __init__(self, config):
        super(Classifier, self).__init__()
        encoder = ResNet18(**config['network'])
        output_feature_dim = encoder.projetion.net[0].in_features
        self.backbone = torch.nn.Sequential(*list(encoder.children())[:-1])
        self.head = LogisticRegression(output_feature_dim, 10)

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = torch.mean(features, dim=[2, 3])
        results = self.head(features)
        return results
            
config = yaml.load(open("../config/config.yaml", "r"), Loader=yaml.FullLoader)
data_transforms = torchvision.transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.STL10('../datasets/', split='train', download=False, transform=data_transforms)
test_dataset = datasets.STL10('../datasets/', split='test', download=False, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=0, drop_last=False, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0, drop_last=False, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = Classifier(config).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
eval_every_n_epochs = 10

for epoch in range(200):
      classifier.train()
      for x, y in train_loader:
          x = x.to(device)
          y = y.to(device)
          optimizer.zero_grad()  
          logits = classifier(x)
          predictions = torch.argmax(logits, dim=1)
          loss = criterion(logits, y)
          loss.backward()
          optimizer.step()
      total = 0
      if epoch % eval_every_n_epochs == 0:
          correct = 0
          classifier.eval()
          for x, y in test_loader:
              x = x.to(device)
              y = y.to(device)
              logits = classifier(x)
              predictions = torch.argmax(logits, dim=1)
              total += y.size(0)
              correct += (predictions == y).sum().item()
          acc = 100 * correct / total
          print(f"Testing accuracy: {np.mean(acc)}")
          classifier.train()