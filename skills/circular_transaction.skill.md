```meta
id: circular_transaction
version: "1.0"
severity: CRITICAL
tags: [circular, transaction, cycle]
enabled: true
```

## Description
Detects circular fund flows between entities, which is a strong indicator of
money laundering through round-tripping transactions.

```pattern
edge_types: [TRANSFER, LOAN, PAYMENT]
```

```python
def detect(G, ctx):
    leads = []
    seen_cycles = set()
    for edge_type in ("TRANSFER", "LOAN", "PAYMENT"):
        cycles = ctx.find_cycles(G, edge_type)
        edge_label = edge_type.lower()
        for cycle in cycles:
            key = frozenset(cycle)
            if key in seen_cycles:
                continue
            seen_cycles.add(key)
            total_amount = 0.0
            evidence = []
            for i, node in enumerate(cycle):
                next_node = cycle[(i + 1) % len(cycle)]
                for u, v, data in ctx.edges_by_type(G, edge_type):
                    if u == node and v == next_node:
                        amount = ctx.parse_amount(str(data.get("amount", "0")))
                        total_amount += amount
                        evidence.append(
                            f"{node} -> {next_node} ({edge_type}, {data.get('amount', 'N/A')})"
                        )
                        break
            score = min(100, 70 + len(cycle) * 5)
            leads.append(ctx.lead(
                title=f"Circular {edge_label} detected ({len(cycle)} entities)",
                severity="CRITICAL",
                score=score,
                entities=cycle,
                evidence=evidence,
                actions=[
                    "Freeze involved accounts pending investigation",
                    "File Suspicious Activity Report (SAR)",
                    "Trace ultimate source and destination of funds",
                ],
            ))
    return leads
```

```test_cases
- description: No cycle
  graph:
    nodes:
      - {id: A, type: Company}
      - {id: B, type: Company}
    edges:
      - {source: A, target: B, type: TRANSFER, amount: "100"}
  expected_leads: 0

- description: Simple 3-node cycle
  graph:
    nodes:
      - {id: A, type: Company}
      - {id: B, type: Company}
      - {id: C, type: Company}
    edges:
      - {source: A, target: B, type: TRANSFER, amount: "500k"}
      - {source: B, target: C, type: TRANSFER, amount: "490k"}
      - {source: C, target: A, type: TRANSFER, amount: "480k"}
  expected_leads: 1
```
