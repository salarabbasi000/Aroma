import openai
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Initialize Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Example LLM-Generated Generic Ingredients
generic_ingredients = ["flour", "sugar", "eggs", "butter", "milk"]

# Step 3: Example Amazon Fresh Products (Ideally Scraped or from API)
amazon_products = [
    "King Arthur Organic All-Purpose Flour",
    "Domino Granulated White Sugar",
    "Organic Brown Eggs - Large",
    "Land O'Lakes Unsalted Butter",
    "Horizon Organic Whole Milk"
]

# Step 4: Convert Amazon Products to Vectors
product_embeddings = model.encode(amazon_products)
dimension = product_embeddings.shape[1]

# Step 5: Store Vectors in FAISS Index
index = faiss.IndexFlatL2(dimension)
index.add(np.array(product_embeddings))

# Step 6: Function to Find Closest Product for an Ingredient
def find_closest_product(ingredient):
    ingredient_vector = model.encode([ingredient])
    _, indices = index.search(np.array(ingredient_vector), k=1)  # Retrieve top match
    return amazon_products[indices[0][0]]

# Step 7: Match Ingredients to Amazon Products
for ingredient in generic_ingredients:
    matched_product = find_closest_product(ingredient)
    print(f"{ingredient} -> {matched_product}")
