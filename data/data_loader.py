import os

class KnowledgeBase:
    def __init__(self, kb_dir: str) -> None:
        assert os.path.exists(kb_dir), f'Path "{kb_dir}" doesn\'t exists.'
        self.dir = kb_dir # absolute path to the dataset

        with open(os.path.join(self.dir, 'entity2id.txt'), 'r') as file:
            self.entity_num = eval(file.readline())
        with open(os.path.join(self.dir, 'relation2id.txt'), 'r') as file:
            self.relation_num = eval(file.readline())
        with open(os.path.join(self.dir, 'train2id.txt'), 'r') as file:
            self.train_num = eval(file.readline())
        with open(os.path.join(self.dir, 'test2id.txt'), 'r') as file:
            self.test_num = eval(file.readline())
        with open(os.path.join(self.dir, 'valid2id.txt'), 'r') as file:
            self.valid_num = eval(file.readline())

    def train_data(self):
        with open(os.path.join(self.dir, 'train2id.txt'), 'r') as file:
            file.readline() # skip the first line which says size of data
            for line in file.readlines():
                yield [eval(i) for i in line.split()]

    def test_data(self):
        with open(os.path.join(self.dir, 'test2id.txt'), 'r') as file:
            file.readline() # skip the first line which says size of data
            for line in file.readlines():
                yield [eval(i) for i in line.split()]

    def valid_data(self):
        with open(os.path.join(self.dir, 'valid2id.txt'), 'r') as file:
            file.readline() # skip the first line which says size of data
            for line in file.readlines():
                yield [eval(i) for i in line.split()]

def get_dataset(data_name):
    # get absolute path of the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, data_name)

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