```meta
id: ownership_chain
version: "1.0"
severity: HIGH
tags: [ownership, chain, depth]
enabled: true
```

## Description
Detects entities that are connected through unusually deep ownership chains,
which may indicate layered shell-company structures designed to obscure the
ultimate beneficial owner.

```pattern
node_types: [Company, Person]
edge_types: [OWNS, CONTROLS]
```

```python
def detect(G, ctx):
    leads = []
    DEPTH_THRESHOLD = 4
    visited = set()
    for company in ctx.nodes_by_type(G, "Company"):
        if company in visited:
            continue
        # Walk ownership chain depth-first
        chain = []
        stack = [(company, [company])]
        while stack:
            node, path = stack.pop()
            owners = ctx.in_neighbors(G, node, "OWNS")
            controllers = ctx.in_neighbors(G, node, "CONTROLS")
            parents = owners + controllers
            if not parents:
                if len(path) >= DEPTH_THRESHOLD:
                    chain = path
                continue
            for parent in parents:
                if parent not in path:
                    stack.append((parent, path + [parent]))
        if len(chain) >= DEPTH_THRESHOLD:
            leads.append(ctx.lead(
                title="Deep ownership chain detected",
                severity="HIGH",
                score=min(100, 50 + len(chain) * 10),
                entities=chain,
                evidence=[
                    "Ownership chain depth: " + str(len(chain)),
                    "Chain: " + " -> ".join(chain),
                ],
                actions=[
                    "Verify ultimate beneficial owner",
                    "Request ownership structure documentation",
                ],
            ))
            visited.update(chain)
    return leads
```

```test_cases
- description: No deep chain
  graph:
    nodes:
      - {id: A, type: Company}
      - {id: B, type: Company}
    edges:
      - {source: B, target: A, type: OWNS}
  expected_leads: 0

- description: Deep chain of 5
  graph:
    nodes:
      - {id: A, type: Company}
      - {id: B, type: Company}
      - {id: C, type: Company}
      - {id: D, type: Company}
      - {id: E, type: Company}
    edges:
      - {source: B, target: A, type: OWNS}
      - {source: C, target: B, type: OWNS}
      - {source: D, target: C, type: OWNS}
      - {source: E, target: D, type: OWNS}
  expected_leads: 1
```
