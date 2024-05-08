from model import VAT, TransE, VATLight

import fire

def main(model: str, data: str, epochs=100, mode: tuple=None, checkpoint: int=None, save=True):
    assert model in ["VAT", "TransE", "VATLight"]
    data = data.lower()

    match model:
        case "VAT":
            model = VAT(data)
        case "TransE":
            model = TransE(data)
        case "VATLight":
            if data == "fb15k":
                model = VATLight(data, './model/checkpoints/TransE-fb15k/20000.pkl')
            else:
                model = VATLight(data, f'./model/checkpoints/TransE-{data}/11500.pkl')

    if checkpoint is not None:
        model.load_parameters(checkpoint)

    if mode is None:
        mode = (20, 0.006, 0)
    model.train(epochs, *mode, save)


if __name__ == "__main__":
    fire.Fire(main)