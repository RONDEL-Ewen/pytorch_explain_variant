import unittest

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath('C:/Users/ewenr/Desktop/02 - PoliTo/02 - Second Semester/02 - Explainable & Trustworthy AI/03 - Project/02 - Code/pytorch_explain_variant'))
from torch_explain.nn.concepts import ConceptEmbedding, ConceptReasoningLayer
from torch_explain.datasets.celebA import celebA

def train_concept_bottleneck_model(x, c, y, embedding_size=1, concepts_name=None, classes_names=None):
    n_concepts = c.shape[1]
    x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)

    y_train = F.one_hot(y_train.long().ravel()).float()
    y_test = F.one_hot(y_test.long().ravel()).float()

    # Encoder
    #encoder = models.resnet34(pretrained=True)
    encoder = models.resnet18(pretrained=True)
    for param in encoder.parameters():
        param.requires_grad = False
    fc = nn.Linear(encoder.fc.in_features, 16)
    encoder.fc = fc

    # Concept Embedding
    concept_embedder = ConceptEmbedding(16, n_concepts, embedding_size)

    # Concept Reasoning
    task_predictor = ConceptReasoningLayer(embedding_size, y_train.shape[1], n_concepts)

    # Full model
    model = torch.nn.Sequential(encoder, concept_embedder, task_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form_c = torch.nn.BCELoss()
    loss_form_y = torch.nn.BCELoss()
    model.train()

    for epoch in range (50):
        optimizer.zero_grad()

        h = encoder(x_train)
        c_emb, c_pred = concept_embedder.forward(h, [0,1], c_train, train=True)
        y_pred = task_predictor.forward(c_emb)

        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        loss = concept_loss + 0.5*task_loss
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            h = encoder(x_test)
            c_emb, c_pred = concept_embedder.forward(h, [0,1], c_test, train=False)
            y_pred = task_predictor.forward(c_emb)

            task_accuracy = accuracy_score(y_test, y_pred > 0.5)
            concept_accuracy = accuracy_score(c_test, c_pred > 0.5)
            print(f'Epoch {epoch}: loss {loss:.4f} task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')

    global_explanations = task_predictor.explain(c_emb, c_pred, concepts_name, classes_names)
    for pred_class in global_explanations:
        print(f"\nClass:\t\t{pred_class['class']}\nExplanation:\t{pred_class['explanation']}")

    return model


class TestTemplateObject(unittest.TestCase):

    def test_deep_core(self):

        print("\n\nCelebA dataset\n")
        x, c, y, concepts_name = celebA(100)
        train_concept_bottleneck_model(x, c, y, embedding_size=16, concepts_name=concepts_name, classes_names=['Male', 'Female'])

        return
    

if __name__ == '__main__':
    unittest.main()