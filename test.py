import networkx as nx
from main import cross_semantic_search, edge_context_text, GraphArtifacts, LeadGenConfig

topology = "- - Person who Inherited money from others  "
edge_texts = [
    "Son got money from father",
    "node=Media - Hubertus von Baumbach Succession\nMedia: Obituary\nWho is Johannes von Baumbach, the world's youngest billionaire? What we know about him| Business News | attrs=Name=Media - Hubertus von Baumbach Succession\nMedia: Obituary\nWho is Johannes von Baumbach, the world's youngest billionaire? What we know about him| Business News; Node Type=News | out_edge_types= | in_edge_types= | out_neighbors= | in_neighbors="
    ]
# Test cross_semantic_search

results = cross_semantic_search(topology, edge_texts, top_k=10)
print("Results:", results)
