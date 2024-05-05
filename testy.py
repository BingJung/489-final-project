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




'''
outtake:
'''
# # compute errors
# encoded_error = [
#     [disturbed[i][d] - translated[i][d] for d in range(self.latent_dim)]
#     for i in range(3)
# ]
# decoded_error = [ # TODO: this can be improved
#     [data[0][d] - decoded[0][d] for d in range(self.entity_dim)],
#     [data[1][d] - decoded[1][d] for d in range(self.relation_dim)],
#     [data[2][d] - decoded[2][d] for d in range(self.entity_dim)]
# ]
# noise_error = [ # TODO: this can be double checked
#     sum([1 / (disturbed[i][d] - translated[i][d]) * norm_noises[i][d] for d in range(self.latent_dim)]) / self.latent_dim
#     for i in range(3)
# ]
# total_error["encoder"] += sum([sum(i) for i in encoded_error])
# total_error["decoder"] += sum([sum(i) for i in decoded_error])
# total_error["noise_gen"] += sum(noise_error)
