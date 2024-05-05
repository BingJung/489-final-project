from model import VAT, TransE

import fire

def main(model: str, data: str, epochs=100, batch_size: int = 20, lr=0.01, momentum=0, checkpoint: int=None, save=True):
    assert model in ["VAT", "TransE"]

    if model == "VAT":
        model = VAT(data)
    else: # "TransE"
        model = TransE(data)

    if checkpoint is not None:
        model.load_parameters(batch_size, lr, momentum, checkpoint)

    model.train(epochs = epochs, batch_size = batch_size, lr = lr, momentum = momentum, save = save)


if __name__ == "__main__":
    fire.Fire(main)