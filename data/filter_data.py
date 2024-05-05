import os, json
from os.path import join
from tqdm import tqdm
import fire # this package needs to be installed

def main(dataset: str, entity_num: int):
    print(f"start filtering {dataset}")
    
    # make directories
    script_dir = os.path.dirname(__file__)
    dataset = dataset.upper()
    orig_dir = join(script_dir, dataset)
    assert os.path.exists(orig_dir), "dataset does not exist"
    new_dir = join(script_dir, f"{dataset}_filtered")
    os.makedirs(new_dir, exist_ok=True)

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

    # filter samples and record relations
    relations = set()
    files = ['train2id.txt', 'test2id.txt', 'valid2id.txt']
    for file_name in files:
        new_file = open(join(new_dir, file_name), 'w')
        with open(join(orig_dir, file_name), 'r') as file:
            file.readline()
            for line in tqdm(file.readlines()):
                ids = [eval(i) for i in line.split()]
                if ids[0] >= entity_num or ids[1] >= entity_num:
                    continue
                if file_name != 'train2id.txt' and ids[2] not in relations:
                    continue
                else:
                    relations.add(ids[2])
                new_file.write(line)
        new_file.close()

    # filter relations
    relation_map = {}
    new_file = open(join(new_dir, 'relation2id.txt'), 'w')
    with open(join(orig_dir, 'relation2id.txt'), 'r') as file:
        file.readline()
        line_id = 0
        for line in tqdm(file.readlines()):
            id = eval(line.split()[-1])
            if id not in relations:
                continue
            new_file.write(line)
            relation_map[id] = line_id
            line_id += 1
    new_file.close()
    
    with open(join(new_dir, 'relation_map.json'), 'w') as file:
        json.dump(relation_map, file)

    # write numbers of lines at beginning of each file
    files.append('entity2id.txt')
    files.append('relation2id.txt')
    for file_name in files:
        with open(join(new_dir, file_name), 'r+') as new_file:
            lines = new_file.readlines()
            new_file.seek(0)
            new_file.write(f"{len(lines)}\n")
            for line in lines:
                new_file.write(line)
            print(file_name, len(lines))
    

if __name__ == "__main__":
    fire.Fire(main)