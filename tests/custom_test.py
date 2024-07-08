import torch
import torch_explain as te
from torch_explain import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_explain.nn.concepts import ConceptReasoningLayer
import torch.nn.functional as F
import numpy as np

def custom_trig(size, random_state=42):
    np.random.seed(random_state)
    h = np.random.normal(0, 2, (size, 7))
    x, y, z, w, a, b, c = h[:, 0], h[:, 1], h[:, 2], h[:, 3], h[:, 4], h[:, 5], h[:, 6]

    # raw features
    input_features = np.stack([
        np.sin(x) + x,
        np.cos(x) + x,
        np.sin(y) + y,
        np.cos(y) + y,
        np.sin(z) + z,
        np.cos(z) + z,
        np.sin(w) + w,
        np.cos(w) + w,
        np.sin(a) + a,
        np.cos(a) + a,
        np.sin(b) + b,
        np.cos(b) + b,
        np.sin(c) + c,
        np.cos(c) + c,
        x ** 2 + y ** 2 + z ** 2 + w ** 2 + a ** 2 + b ** 2 + c ** 2,
    ]).T

    # concetps
    concetps = np.stack([
        x > 0,
        y > 0,
        z > 0,
    ]).T

    # task
    downstream_task = (x + y + z + w + a + b + c) > 1

    input_features = torch.FloatTensor(input_features)
    concetps = torch.FloatTensor(concetps)
    downstream_task = torch.FloatTensor(downstream_task)
    return input_features, concetps, downstream_task.unsqueeze(-1)

x, c, y = custom_trig(500)
x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)
# print only first 5 entries of x_train and c_train
print(x_train[:5])
print(c_train[:5])
embedding_size = 2
concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 10),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
)

y_train = F.one_hot(y_train.long().ravel()).float()
y_test = F.one_hot(y_test.long().ravel()).float()

task_predictor = ConceptReasoningLayer(embedding_size, y_train.shape[1])
model = torch.nn.Sequential(concept_encoder, task_predictor)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_form = torch.nn.BCELoss()
model.train()
for epoch in range(501):
    if epoch % 100 == 0:
        print(f'Epoch {epoch}...')
    optimizer.zero_grad()

    # generate concept and task predictions
    c_emb, c_pred = concept_encoder(x_train)
    y_pred = task_predictor(c_emb, c_pred)

    # compute loss
    concept_loss = loss_form(c_pred, c_train)
    task_loss = loss_form(y_pred, y_train)
    loss = concept_loss + 0.5*task_loss

    loss.backward()
    optimizer.step()


# switch to evaluation mode
model.eval()
with torch.no_grad():
    # generate concept and task predictions for test set
    c_emb, c_pred = concept_encoder(x_test)
    y_pred = task_predictor(c_emb, c_pred)

    # calculate test accuracy
    y_pred_classes = y_pred.argmax(dim=1)
    y_test_classes = y_test.argmax(dim=1)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

local_explanations = task_predictor.explain(c_emb, c_pred, 'local')
global_explanations = task_predictor.explain(c_emb, c_pred, 'global')
#print(local_explanations)
print(global_explanations)
