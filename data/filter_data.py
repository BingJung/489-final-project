import os, shutil
from os.path import join
from tqdm import tqdm
import fire # this package needs to be installed

def main(dataset: str, entity_num: int):
    # make directories
    dataset = dataset.upper()
    orig_dir = join('.', dataset)
    assert os.path.exists(orig_dir)
    new_dir = join('.', f"{dataset}_filtered")
    os.makedirs(new_dir, exist_ok=True)

    # copy relation2id
    shutil.copy(join(orig_dir, 'relation2id.txt'), new_dir)

    # filter entity2id
    new_file = open(join(new_dir, 'entity2id.txt'), 'w')
    with open(join(orig_dir, 'entity2id.txt'), 'r') as file:
        file.readline()
        for line in tqdm(file.readlines()):
            id = eval(line.split()[-1])
            if id >= entity_num:
                continue
            new_file.write(line)
    new_file.close()

    # filter samples
    files = ['train2id.txt', 'test2id.txt', 'valid2id.txt']
    for file_name in files:
        new_file = open(join(new_dir, file_name), 'w')
        with open(join(orig_dir, file_name), 'r') as file:
            file.readline()
            for line in tqdm(file.readlines()):
                ids = [eval(i) for i in line.split()]
                if ids[0] >= entity_num or ids[1] >= entity_num:
                    continue
                new_file.write(line)
        new_file.close()

    # write numbers of lines
    files.append('entity2id.txt')
    for file_name in files:
        with open(join(new_dir, file_name), 'r+') as new_file:
            lines = new_file.readlines()
            new_file.seek(0)
            new_file.write(f"{len(lines)}\n")
            for line in lines:
                new_file.write(line)
    

if __name__ == "__main__":
    fire.Fire(main)