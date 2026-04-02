import argparse
import json
import sys

from main import (
    LeadGenConfig,
    generate_targets,
    init_llm_client,
    load_artifacts,
)
from analyze_node import analyze_node


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = LeadGenConfig(
        debug=args.debug,
        auto_write_rules=args.write_rules,
        score_threshold=args.threshold,
        llm_batch_size=args.batch_size,
    )
    targets, new_rules = generate_targets(
        excel_path=args.excel,
        final_goal=args.goal,
        topology_path=args.topology,
        config=cfg,
    )

    print(f"\n=== High-Priority Targets ({len(targets)}) ===")
    for t in targets[: args.top]:
        print(f"  {t['node_name']} | {t['score']} | {t['reason']}")

    print(f"\n=== Discovered Rules ({len(new_rules)}) ===")
    for rule in new_rules:
        print(f"  - {rule}")

    if args.output:
        payload = {"targets": targets, "new_rules": new_rules}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_analyze(args: argparse.Namespace) -> None:
    cfg = LeadGenConfig(debug=args.debug)
    artifacts = load_artifacts(args.excel, args.topology, cfg)
    llm = init_llm_client()

    result = analyze_node(
        node=args.node,
        artifacts=artifacts,
        final_goal=args.goal,
        config=cfg,
        llm=llm,
        score_reason=args.reason,
    )

    print(f"\n=== Analysis: {args.node} ===")
    print(f"\nInsight:\n{result['insight']}")
    print(f"\nRecommended Action:\n{result['recommended_action']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output}")


def cmd_list_nodes(args: argparse.Namespace) -> None:
    cfg = LeadGenConfig(debug=False)
    artifacts = load_artifacts(args.excel, args.topology, cfg)

    nodes = sorted(artifacts.node_names)
    print(f"\n=== Nodes ({len(nodes)}) ===")
    for node in nodes:
        print(f"  {node}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lead-gen",
        description="Lead generation CLI — target discovery and node analysis",
    )
    parser.add_argument("--excel", default="COI_Template.xlsx", help="Excel data file (default: COI_Template.xlsx)")
    parser.add_argument("--topology", default="topology.md", help="Topology rules file (default: topology.md)")
    parser.add_argument("--goal", default="Identify high-potential leads for financial services based on their relationships and attributes in the graph.")
    parser.add_argument("--debug", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="Run target generation + rule discovery")
    gen.add_argument("--threshold", type=int, default=60, help="Score threshold (default: 60)")
    gen.add_argument("--batch-size", type=int, default=20, help="LLM batch size (default: 20)")
    gen.add_argument("--top", type=int, default=10, help="Number of top targets to print (default: 10)")
    gen.add_argument("--write-rules", action="store_true", help="Append discovered rules to topology file")
    gen.add_argument("--output", "-o", help="Save JSON results to file")

    # analyze
    ana = sub.add_parser("analyze", help="Analyze a single node")
    ana.add_argument("node", help="Node name to analyze")
    ana.add_argument("--reason", default="", help="Prior scoring reason for context")
    ana.add_argument("--output", "-o", help="Save JSON result to file")

    # list-nodes
    sub.add_parser("list-nodes", help="List all node names in the graph")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "generate": cmd_generate,
        "analyze": cmd_analyze,
        "list-nodes": cmd_list_nodes,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
