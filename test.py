from model import VAT, TransE

import fire

def main(model: str, data: str, mode: tuple):
    assert model in ["VAT", "TransE"]

    if model == "VAT":
        model = VAT(data)
    else: # "TransE"
        model = TransE(data)

    model.load_parameters(*mode)
    model.test()



if __name__ == "__main__":
    fire.Fire(main)