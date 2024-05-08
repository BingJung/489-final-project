from model import VAT, TransE, VATLight

import fire

def main(model: str, data: str, checkpoint: int, mode: tuple=()):
    assert model in ["VAT", "TransE", "VATLight"]            

    match model:
        case "VAT":
            model = VAT(data)
        case "TransE":
            model = TransE(data)
        case "VATLight":
            model = VATLight(data, f'./model/checkpoints/TransE-fb15k/20000.pkl')

    model.load_parameters(checkpoint, *mode)
    model.test()

if __name__ == "__main__":
    fire.Fire(main)