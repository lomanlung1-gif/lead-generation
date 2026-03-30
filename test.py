from sentence_transformers.cross_encoder import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

topology = "Individuals selling their company should be rich."
edge_types = ["Works For",
"Shareholder",
"Business Ownership"]


# Cross-Encoder 直接对每一对打分
pairs = [[topology, et] for et in edge_types]
scores = model.predict(pairs)

for et, score in zip(edge_types, scores):
    print(f"{et}: {score:.4f}")
# "Sales of company" 应获得最高分