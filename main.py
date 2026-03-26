import os
import json
from pathlib import Path
import pandas as pd
import networkx as nx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".secrets" / "deepseek.env")
load_dotenv(project_root / ".env")


class GraphRAGLeadGenerator:
    def __init__(
        self,
        excel_path: str,
        final_goal: str,
        topology_path: str = "topology.md",
        node_sheet: str = "Node",
        edge_sheet: str = "Edge",
    ):
        self.final_goal = final_goal
        self.topology_path = topology_path

        # Load seed topology rules from file
        self.seed_topology = self._load_topology()
        print(f"📖 Loaded topology rules from {self.topology_path}")

        # Load graph data from Excel
        print("📥 Loading graph data...")
        df_nodes = pd.read_excel(excel_path, sheet_name=node_sheet)
        df_edges = pd.read_excel(excel_path, sheet_name=edge_sheet)

        self.G = nx.DiGraph()
        for _, row in df_nodes.iterrows():
            self.G.add_node(row["Name"], **row.to_dict())
        for _, row in df_edges.iterrows():
            self.G.add_edge(
                row["Source Node"],
                row["Target Node"],
                edge_type=row["Edge Type"],
                edge_info=row.get("Other Info Object (JSON)", ""),
            )

        # Initialize semantic retrieval model and precompute embeddings
        print("🔧 Initializing semantic retrieval model...")
        self.bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

        self.node_names = list(self.G.nodes)
        node_texts = [f"Node: {n}, attrs: {self.G.nodes[n]}" for n in self.node_names]
        self.node_embeddings = self.bi_encoder.encode(node_texts, convert_to_tensor=True)

        self.edge_list = []
        edge_texts = []
        for u, v, attr in self.G.edges(data=True):
            edge_texts.append(f"Relation: {u} {attr['edge_type']} {v}, Info: {attr['edge_info']}")
            self.edge_list.append((u, v, attr))
        self.edge_embeddings = self.bi_encoder.encode(edge_texts, convert_to_tensor=True)

        # Initialize LLM client
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        print("✅ Initialization complete!")

    def _load_topology(self) -> str:
        if not os.path.exists(self.topology_path):
            raise FileNotFoundError(f"Topology file not found: {self.topology_path}")
        with open(self.topology_path, "r", encoding="utf-8") as f:
            return f.read()

    def _append_rules_to_topology(self, new_rules: list) -> int:
        current_content = self._load_topology()
        truly_new = [r for r in new_rules if r not in current_content]
        if not truly_new:
            return 0
        with open(self.topology_path, "a", encoding="utf-8") as f:
            for rule in truly_new:
                f.write(f"\n- {rule}")
        self.seed_topology = self._load_topology()
        return len(truly_new)

    def _retrieve_relevant_context(self, query: str | None = None):
        if query is None:
            query = f"{self.seed_topology} {self.final_goal}"

        q_emb = self.bi_encoder.encode(query, convert_to_tensor=True)

        node_hits = util.semantic_search(q_emb, self.node_embeddings, top_k=30)[0]
        top_nodes = [self.node_names[hit["corpus_id"]] for hit in node_hits]

        edge_hits = util.semantic_search(q_emb, self.edge_embeddings, top_k=30)[0]
        top_edges = [self.edge_list[hit["corpus_id"]] for hit in edge_hits]

        # Build deduplicated context from top edges + 1-hop neighbors
        context = []
        seen = set()
        for u, v, attr in top_edges:
            context.append(f"{u} {attr['edge_type']} {v}")
            seen.add((u, v))

        for node in top_nodes:
            for neighbor in list(self.G.neighbors(node)) + list(self.G.predecessors(node)):
                pair = (node, neighbor) if self.G.has_edge(node, neighbor) else (
                    (neighbor, node) if self.G.has_edge(neighbor, node) else None
                )
                if pair and pair not in seen:
                    edge = self.G.get_edge_data(*pair)
                    if edge:
                        context.append(f"{pair[0]} {edge['edge_type']} {pair[1]}")
                        seen.add(pair)

        return top_nodes, context

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # remove closing fence
            stripped = "\n".join(lines)
        return stripped

    def generate_targets(self):
        top_nodes, _ = self._retrieve_relevant_context()

        # Build node data for LLM analysis
        nodes_to_process = []
        for node in top_nodes:
            relations = [
                f"{node} {self.G[node][nb]['edge_type']} {nb}"
                for nb in self.G.neighbors(node)
            ]
            nodes_to_process.append({
                "node": node,
                "attrs": self.G.nodes[node],
                "relations": relations,
            })

        if not nodes_to_process:
            return []

        print(f"🤖 LLM analyzing {len(nodes_to_process)} candidate nodes...")
        prompt = f"""You are an expert lead-scoring analyst for financial services.

GOAL: {self.final_goal}

TOPOLOGY RULES (proven heuristics — apply these strictly):
{self.seed_topology}

NODE DATA:
{json.dumps(nodes_to_process, ensure_ascii=False, indent=2, default=str)}

INSTRUCTIONS:
1. Score each node 0-100 on how well it matches the goal, using ONLY the topology rules and node data above. Do NOT hallucinate attributes not present in the data.
2. For every node scored above 30, provide a concise reason tied to a specific rule or data point.
3. Identify any new, generalizable topology rules you can confidently derive from this data that are NOT already listed above. Only include rules you are certain about.

Respond with ONLY valid JSON (no markdown, no commentary):
{{
  "targets": [
    {{"node_name": "<exact node name>", "score": <int 0-100>, "reason": "<concise justification>"}}
  ],
  "new_discovered_rules": [
    "<new rule 1>",
    "<new rule 2>"
  ]
}}"""

        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = self._strip_code_fences(resp.choices[0].message.content)
            res = json.loads(raw)

            results = [t for t in res.get("targets", []) if t.get("score", 0) > 30]

            new_rules = res.get("new_discovered_rules", [])
            if new_rules:
                added = self._append_rules_to_topology(new_rules)
                if added:
                    print(f"🎉 Discovered {added} new topology rules — saved to {self.topology_path}")
                    for r in new_rules[:added]:
                        print(f"   - {r}")

        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            results = []

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def ask(self, question: str):
        _, context = self._retrieve_relevant_context(query=question)
        resp = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": (
                    "Answer the following question using ONLY the provided context. "
                    "Do not fabricate information. If the context is insufficient, say so.\n\n"
                    f"Question: {question}\n\nContext:\n" + "\n".join(context)
                ),
            }],
        )
        return resp.choices[0].message.content


if __name__ == "__main__":
    generator = GraphRAGLeadGenerator(
        excel_path="COI_Template.xlsx",
        final_goal="Increase financial product sales revenue by identifying high-value prospective clients",
        topology_path="topology.md",
    )

    targets = generator.generate_targets()

    print("\n=== High-Priority Target List ===")
    for t in targets[:10]:
        print(f"Node: {t['node_name']} | Score: {t['score']} | Reason: {t['reason']}")