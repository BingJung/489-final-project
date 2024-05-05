from data import get_dataset, KnowledgeBase
from model import VAT

model = VAT('WN18RR')
model.train(save=True)
