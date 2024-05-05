from data import get_dataset, KnowledgeBase
from model import VAT, TransE

import fire

def main(model: str, data: str, save=True):
    assert model in ["VAT", "TransE"]

    if model == "VAT":
        model = VAT(data)
    else: # "TransE"
        model = TransE(data)

    model.train(save = save)


if __name__ == "__main__":
    fire.Fire(main)