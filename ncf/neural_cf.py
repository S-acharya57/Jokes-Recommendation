import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn


jokes_df = pd.read_csv('../data/jester_items.csv')
ratings_df = pd.read_csv('../data/jester_ratings.csv')

df = ratings_df 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

df['userId'] = df['userId'].astype('category').cat.codes.values
df['jokeId'] = df['jokeId'].astype('category').cat.codes.values

# unique number of users and jokes
num_users = 59132
num_jokes = 140 


class NeuralCollaborative(nn.Module):
    
    def __init__(self, num_users, num_jokes, embedding_dim=256):
        super(NeuralCollaborative, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.joke_embedding = nn.Embedding(num_jokes, embedding_dim)

        # layers
        self.fc1 = nn.Linear(embedding_dim*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1) # single value of prediction rating

    def forward(self, user_id, joke_id):
        user_embedding = self.user_embedding(user_id)
        joke_embedding = self.joke_embedding(joke_id)

        # concatenate user and joke embeddings
        x = torch.cat([user_embedding, joke_embedding], dim=1)

        # through the fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        output = self.fc4(x)
        return output 
    

embedding_dim = 256

model = NeuralCollaborative(num_users, num_jokes, embedding_dim)
model = model.to(device)
model = nn.DataParallel(model, device_ids=[0])

# loading pretrained model
model_path = 'ncf_trained_50.pth'

'''

# using DataParallel in Kaggle
# so removing word 'module' from it
state_dict = torch.load(model_path)


# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)


'''

model.load_state_dict(torch.load(model_path))
def infer(user_id, joke_id):
    
    with torch.no_grad():
        user_id_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
        joke_id_tensor = torch.tensor([joke_id], dtype=torch.long).to(device)
        prediction = model(user_id_tensor, joke_id_tensor)
        return prediction.item()
    

user_id = int(input("Enter User_id"))
joke_id = int(input("Enter joke id"))

print(infer(user_id=user_id,joke_id=joke_id))