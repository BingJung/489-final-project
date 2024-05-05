# from data import get_dataset
# fb15k237 = get_dataset("FB15K237")

# for line in fb15k237.train_data():
#     print(line)
#     break



from model.AutoTranscoder import get_dims

print(get_dims(15451, 128, 4))



# l = [1, 2, 3, 4]

# def get(l):
#     for i in l: 
#         yield i

# haha = get(l)

# for i in haha:
#     print(i)



