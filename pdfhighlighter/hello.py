import os
import fitz  
import nltk
import re
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import LongformerModel, LongformerTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces

nltk.download('punkt')

# Load SBERT for sentence embeddings
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load Longformer for context-aware embeddings
longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

# Clean and normalize text
def cleanText(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\b(?:introduction|abstract|conclusion|references)\b', '', text, flags=re.IGNORECASE)  
    return text

# Extract text from PDF
def extractText(pdfPath):
    doc = fitz.open(pdfPath)
    allText = []
    for page in doc:
        blocks = page.get_text("blocks")
        pageText = " ".join([b[4] for b in blocks if len(b[4].split()) > 5])  
        allText.append(pageText)
    return allText  

# Split text into sentences
def splitSentences(text):
    return sent_tokenize(text)

# Generate embeddings for sentences
def getSentenceEmbeddings(sentences):
    sbertEmbeddings = sbert.encode(sentences)
    
    longformerEmbeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = longformer(**inputs)
        longformerEmbeddings.append(outputs.last_hidden_state.mean(dim=1).numpy().flatten())

    longformerEmbeddings = np.array(longformerEmbeddings)
    combinedEmbeddings = np.hstack((sbertEmbeddings, longformerEmbeddings))
    return combinedEmbeddings

# Define RL environment
class HighlightSelector(gym.Env):
    def _init_(self, scores):
        super(HighlightSelector, self)._init_()
        self.scores = np.array(scores)
        self.numSentences = len(scores)

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.numSentences,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.numSentences)

        self.state = np.zeros(self.numSentences)
        self.currentStep = 0

    def reset(self):
        self.state = np.zeros(self.numSentences)
        self.currentStep = 0
        return self.state.copy()

    def step(self, action):
        reward = self.scores[action] if self.state[action] == 0 else -1  
        self.state[action] = 1
        self.currentStep += 1
        done = self.currentStep >= self.numSentences
        return self.state.copy(), reward, done, {}

# RL-based selection of sentences
def selectSentences(sentences, scores, maxHighlights=20):
    if len(sentences) < 2:
        return sentences  

    env = DummyVecEnv([lambda: HighlightSelector(scores)])
    ppo = PPO("MlpPolicy", env, verbose=0)

    selectedSentences = []
    state = env.reset()

    for _ in range(maxHighlights):
        action, _ = ppo.predict(state, deterministic=True)
        if state[0][action] == 0:
            selectedSentences.append(sentences[action])
        state, _, done, _ = env.step(action)
        if done:
            break

    return selectedSentences

# Apply highlights in PDF
def applyHighlights(originalPdf, importantSentences, outputPdf):
    doc = fitz.open(originalPdf)

    for page in doc:
        for sentence in importantSentences:
            areas = page.search_for(sentence)
            if not areas:
                words = sentence.split()
                for i in range(len(words) - 4):
                    phrase = " ".join(words[i:i+4])
                    areas = page.search_for(phrase)
                    if areas:
                        break

            for rect in areas:
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=(1, 1, 0))  
                highlight.update()

    doc.save(outputPdf, garbage=4, deflate=True)
    print(f"✅ Highlighted PDF saved as: {outputPdf}")

# Main execution function
def run():
    root = tk.Tk()
    root.withdraw()
    pdfPath = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF Files", ".pdf"), ("All Files", ".*")])

    if not pdfPath:
        print("❌ No PDF selected. Existing.")
        return

    print(f"📄 Selected PDF: {pdfPath}")
    pageTexts = extractText(pdfPath)
    
    allHighlights = []

    for text in pageTexts:
        text = cleanText(text)
        sentences = splitSentences(text)
        if not sentences:
            continue  

        embeddings = getSentenceEmbeddings(sentences)
        scores = cosine_similarity(embeddings).mean(axis=1)

        numToHighlight = min(len(sentences), 20)
        importantSentences = [sentences[i] for i in np.argsort(scores)[::-1][:numToHighlight]]

        rlSelectedSentences = selectSentences(sentences, scores, maxHighlights=numToHighlight)

        finalHighlights = list(set(importantSentences + rlSelectedSentences))
        allHighlights.extend(finalHighlights)

    outputPdf = os.path.splitext(pdfPath)[0] + "_highlighted.pdf"
    applyHighlights(pdfPath, allHighlights, outputPdf)

if __name__ == '__main__':
    run()
