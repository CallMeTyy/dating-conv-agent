from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
from typing import List, Dict



app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dtype = torch.float16 if device.type == "cuda" else torch.float32

model = BertModel.from_pretrained("prajjwal1/bert-tiny", torch_dtype=dtype).to(device)
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
model.eval()


def embed(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)

# Already embed it, so the api is faster
CATEGORIES = {
    "activity": [embed("Sports"), embed("Movie"), embed("Music"), embed("Sightseeing") ],
    "eating": [embed("Food"), embed("Restaurant"), embed("Cafe")],
    # add more categories here that the api user can pick (for other use cases, e.g. fancy-clothes, chill-clothes)
}


class SimilarityRequest(BaseModel):
    text: str
    categories: List[str]

class SimilarityResponse(BaseModel):
    category: str
    similarity: float

@app.post("/most_similar", response_model=SimilarityResponse)
def most_similar(req: SimilarityRequest):
    user_embedding = embed(req.text)
    best_cat = None
    best_sim = -1.0
    for cat in [cat for cat in req.categories if cat in CATEGORIES]:
        for kw_emb in CATEGORIES[cat]:
            sim = torch.cosine_similarity(user_embedding, kw_emb).item()
            if sim > best_sim:
                best_sim = sim
                best_cat = cat
    if best_cat is None:
        raise HTTPException(status_code=404, detail="No valid categories found.")
    return {"category": best_cat, "similarity": best_sim}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)