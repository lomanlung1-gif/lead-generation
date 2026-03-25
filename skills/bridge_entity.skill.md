```meta
id: bridge_entity
version: "1.0"
severity: MEDIUM
tags: [bridge, intermediary, betweenness]
enabled: true
```

## Description
Identifies entities with unusually high betweenness centrality that act as
critical bridges in the network, which may indicate they are being used as
intermediary pass-through entities in complex financial schemes.

```pattern
node_types: [Company, Person, Account]
```

```python
def detect(G, ctx):
    leads = []
    if G.number_of_nodes() < 3:
        return leads
    top_entities = ctx.betweenness(G, top_pct=0.05)
    for entity, score in top_entities:
        if score < 0.2:
            continue
        community = ctx.community_of(G, entity)
        entity_type = ctx.node_attr(G, entity, "type", "unknown")
        leads.append(ctx.lead(
            title=f"High-centrality bridge entity: {entity}",
            severity="MEDIUM",
            score=min(100, int(score * 200)),
            entities=[entity],
            evidence=[
                f"Betweenness centrality: {score:.4f}",
                f"Entity type: {entity_type}",
                f"Community size: {len(community)}",
                f"Connected to {len(list(ctx.in_neighbors(G, entity)))} inbound "
                f"and {len(list(ctx.out_neighbors(G, entity)))} outbound entities",
            ],
            actions=[
                "Review all transactions passing through this entity",
                "Verify legitimate business purpose for central role",
                "Request corporate structure documentation",
            ],
        ))
    return leads
```

```test_cases
- description: Too few nodes — no leads
  graph:
    nodes:
      - {id: A, type: Company}
      - {id: B, type: Company}
    edges: []
  expected_leads: 0

- description: Star topology — centre has high betweenness
  graph:
    nodes:
      - {id: Hub, type: Company}
      - {id: N1, type: Company}
      - {id: N2, type: Company}
      - {id: N3, type: Company}
      - {id: N4, type: Company}
      - {id: N5, type: Company}
    edges:
      - {source: N1, target: Hub, type: TRANSFER}
      - {source: N2, target: Hub, type: TRANSFER}
      - {source: N3, target: Hub, type: TRANSFER}
      - {source: Hub, target: N4, type: TRANSFER}
      - {source: Hub, target: N5, type: TRANSFER}
  expected_leads: 1
```
