from data import get_dataset, KnowledgeBase
from model import VAT

model = VAT('FB15K237')
model.train()
