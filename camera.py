import torch


model = torch.load('5-epochs.pth', weights_only=False)

classes = list('ABCDEFGHIJKLMNOPQRSTUVWYXZ')
classes.append('del')
classes.append('nothing')
classes.append('space')
# model outputs an number 0-28, which corresponds to the index of the label in this classes list.