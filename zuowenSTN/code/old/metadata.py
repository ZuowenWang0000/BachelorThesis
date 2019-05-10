import pickle


with open('mf7SCCN7b3_training_metadata.pkl', 'rb') as f:
    data = pickle.load(f)

for (k,v) in data['checkpoints'].items():
	print(data['checkpoints'][k]['test_adv_loss'])
