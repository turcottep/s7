import torch

class Net(torch.nn.Module):

	def __init__(self):

		super(Net, self).__init__()

		self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.mp1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
		self.rl1 = torch.nn.ReLU()

		self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.mp2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
		self.rl2 = torch.nn.ReLU()

		self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.mp3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
		self.rl3 = torch.nn.ReLU()

		self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.mp4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
		self.rl4 = torch.nn.ReLU()

		self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=10, kernel_size=8, stride=1, padding=0)
		self.sf = torch.nn.Softmax(dim=1)

	def forward(self, x):

		x = self.rl1(self.mp1(self.conv1(x)))
		x = self.rl2(self.mp2(self.conv2(x)))
		x = self.rl3(self.mp3(self.conv3(x)))
		x = self.rl4(self.mp4(self.conv4(x)))

		x = self.sf(self.conv5(x).view(x.size(0), -1))

		return x
