import os, json
from os.path import join

class KnowledgeBase:
    def __init__(self, kb_dir: str) -> None:
        assert os.path.exists(kb_dir), f'Path "{kb_dir}" doesn\'t exists.'
        self.dir = kb_dir # absolute path to the dataset

        with open(join(self.dir, 'entity2id.txt'), 'r') as file:
            self.entity_num = eval(file.readline())
        with open(join(self.dir, 'relation2id.txt'), 'r') as file:
            self.relation_num = eval(file.readline())
        with open(join(self.dir, 'train2id.txt'), 'r') as file:
            self.train_num = eval(file.readline())
        with open(join(self.dir, 'test2id.txt'), 'r') as file:
            self.test_num = eval(file.readline())
        with open(join(self.dir, 'valid2id.txt'), 'r') as file:
            self.valid_num = eval(file.readline())

        if os.path.exists(join(kb_dir, 'relation_map.json')):
            with open(join(kb_dir, 'relation_map.json'), 'r') as file:
                self.relation_map = json.load(file)
        else:
            self.relation_map = None

    def train_data(self):
        '''
        return single data sample once
        format: [h, t, r]
        '''
        with open(join(self.dir, 'train2id.txt'), 'r') as file:
            file.readline() # skip the first line which says size of data
            for line in file.readlines():
                triple = [eval(i) for i in line.split()]
                if self.relation_map is not None:
                    triple[-1] = self.relation_map[str(triple[-1])]
                yield triple

    def test_data(self):
        with open(join(self.dir, 'test2id.txt'), 'r') as file:
            file.readline() # skip the first line which says size of data
            for line in file.readlines():
                triple = [eval(i) for i in line.split()]
                if self.relation_map is not None:
                    triple[-1] = self.relation_map[str(triple[-1])]
                yield triple

    def valid_data(self):
        with open(join(self.dir, 'valid2id.txt'), 'r') as file:
            file.readline() # skip the first line which says size of data
            for line in file.readlines():
                triple = [eval(i) for i in line.split()]
                if self.relation_map is not None:
                    triple[-1] = self.relation_map[str(triple[-1])]
                yield triple

def get_dataset(data_name):
    # get absolute path of the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = join(current_dir, data_name)

    return KnowledgeBase(data_dir)    



# from data.loader import TrainDataLoader, TestDataLoader

# train_dataloader = TrainDataLoader(
# 	in_path = "./data/FB15K237/", 
# 	nbatches = 100,
# 	threads = 8, 
# 	sampling_mode = "normal", 
# 	bern_flag = 1, 
# 	filter_flag = 1, 
# 	neg_ent = 25,
# 	neg_rel = 0)

# test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# for i in TrainDataLoader:
#     print(i)
#     break