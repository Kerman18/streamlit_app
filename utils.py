import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class NextTokenPredictor(nn.Module):
    def __init__(self, vocab_size, emb_size, block_size, hidden_size):
        torch.manual_seed(42)
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size,padding_idx=0)
        self.fc1 = nn.Linear(emb_size * block_size, hidden_size) # first fully connected layer
        #self.fc2=nn.Linear(hidden_size,hidden_size) #second fully connected layer
        self.fc3 = nn.Linear(hidden_size, vocab_size) # output layer
    
    def forward(self, x,act_fun='relu'):
        act = torch.tanh if act_fun == 'tanh' else F.relu
        x = self.emb(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = act(self.fc1(x))
        #x = act(self.fc2(x))
        x = self.fc3(x)
        return x


def clean_text(text):
    text=text.lower()
    text = re.sub(r"(?<=\w)'(?=\w)", "", text) # Remove apostrophes in contractions
    text = re.sub(r"[^a-zA-Z0-9\.]", " ", text) # Keep only alphanumeric characters and periods
    text = re.sub(r"\s+", " ", text) # Replace multiple spaces with a single space
    
    return text

@torch.no_grad()
def generate_next_words(model, word2idx, idx2word, text, act_fun,num_words, temperature, context_length,top_k):
    model.eval()
    cleaned_text = clean_text(text)
    words = cleaned_text.split()

    # Convert tokens to indices (handle unknowns)
    unk_idx = word2idx["<unk>"]
    pad_idx = word2idx[" <pad> "]

    token = [word2idx.get((' '+w), unk_idx) for w in words]
    context=token[-context_length:]

    if len(context) < context_length:
            context = [pad_idx] * (context_length - len(context)) + context

    for _ in range(num_words):
        x = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        logits = model(x,act_fun)

        probs = F.softmax(logits / temperature, dim=1)
        probs[0,unk_idx]=0
        probs[0,pad_idx]=0

        probs_k, idx_k = torch.topk(probs, k=top_k)
        next_id = torch.multinomial(probs_k, num_samples=1).item()
        
        word = idx2word[next_id]
        
        if word in ['<unk>', ' <pad> ']:
            continue
        
        token.append(next_id)

    return " ".join(idx2word.get(i, "<unk>") for i in token)

