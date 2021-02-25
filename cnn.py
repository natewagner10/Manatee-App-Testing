"""
Mote Marine Laboratory Collaboration

Manatee Matching Program

Written by Nate Wagner, Rosa Gradilla 
"""

############################################################
#  Tail mutilation cnn
############################################################

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        """
        Tail mutilation classifier. Inputs 196x196 cropped image of tail.

        Returns
        -------
        int
            classification of containing a tail mutilation or not

        """
        super(CNN,self).__init__()        
        self.layer1 = []
        self.layer1.extend([
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), # 16 @ 196 * 196
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # 16 99 * 99
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # 32 99 * 99
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # 32 50*50
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # 64 50 * 50
            nn.ReLU(),                                     
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), # 128 50 * 50
            nn.ReLU(),                                                 
        ])
        self.layer1 = nn.Sequential(*self.layer1)
        self.fc = []
        self.fc.extend([
            nn.Linear(320000,2),
            nn.Sigmoid(),            
            ])
        self.fc = nn.Sequential(*self.fc)
    def forward(self, z):
        out1 = self.layer1(z)
        out2 = self.fc(out1.view(out1.size(0),-1))        
        return out2
    