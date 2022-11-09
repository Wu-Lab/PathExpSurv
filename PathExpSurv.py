import torch
import torch.nn as nn

class PathExpSurv(nn.Module):
	def __init__(self, In_Nodes, Pathway_Nodes, Pathway_Mask,beta):
		super(PathExpSurv, self).__init__()

		self.pathway_mask = Pathway_Mask
		self.bn_input = nn.BatchNorm1d(In_Nodes)

		self.sc1 = nn.Linear(In_Nodes, Pathway_Nodes,bias=False)
		self.sc2 = nn.Linear(In_Nodes, Pathway_Nodes,bias=False)
		self.bn = nn.BatchNorm1d(Pathway_Nodes)
		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()
		self.sc4 = nn.Linear(Pathway_Nodes + 1, 1, bias=False)
		self.sc4.weight.data.uniform_(-0.001, 0.001)
		self.beta=beta

	def forward(self, x_1, x_2):
		self.sc1.weight.data.clamp_(0)
		self.sc2.weight.data.clamp_(0)
		self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
		self.sc2.weight.data = self.sc2.weight.data.mul(-self.pathway_mask+1)

		x_1 = self.tanh(self.bn(self.sc1(self.bn_input(x_1))+self.beta*self.sc2(self.bn_input(x_1))))
		x_cat = torch.cat((x_1, x_2), 1)
		lin_pred = self.tanh(self.sc4(x_cat))
		
		return lin_pred

