"""Entry point for the compliance graph detection system.

Usage
-----
    python main.py --excel data.xlsx [--skills skills/] [--discover] [--rounds 3]

The script prints a summary of all leads to stdout and exits.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compliance Graph Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--excel",
        required=True,
        help="Path to the Excel workbook (.xlsx)",
    )
    parser.add_argument(
        "--skills",
        default="skills",
        help="Directory containing .skill.md files",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        default=False,
        help="Enable LLM-based skill discovery (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Maximum number of LLM discovery rounds",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use for skill discovery",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write leads as JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ── 1. Build graph ─────────────────────────────────────────────────────
    from graph_builder import GraphBuilder

    logger.info("Loading graph from '%s' …", args.excel)
    try:
        G = GraphBuilder.from_excel(args.excel)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )

    # ── 2. Run detection engine ────────────────────────────────────────────
    from engine.detection_engine import DetectionEngine

    engine = DetectionEngine(
        skills_dir=args.skills,
        discovery_rounds=args.rounds,
        use_discovery=args.discover,
        llm_model=args.model,
    )
    leads = engine.run(G)

    # ── 3. Report leads ────────────────────────────────────────────────────
    if not leads:
        print("No leads found.")
        return 0

    print(f"\n{'='*60}")
    print(f"  Found {len(leads)} lead(s)")
    print(f"{'='*60}\n")

    for i, lead in enumerate(leads, 1):
        print(f"[{i}] {lead.title}")
        print(f"    Skill:    {lead.skill_id}")
        print(f"    Severity: {lead.severity.value}")
        print(f"    Score:    {lead.score:.0f}/100")
        print(f"    Entities: {', '.join(lead.entities)}")
        for ev in lead.evidence:
            print(f"    Evidence: {ev}")
        for act in lead.actions:
            print(f"    Action:   {act}")
        print()

    # ── 4. Optional JSON output ────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        leads_data = [
            {
                "id": lead.id,
                "skill_id": lead.skill_id,
                "title": lead.title,
                "severity": lead.severity.value,
                "score": lead.score,
                "entities": lead.entities,
                "evidence": lead.evidence,
                "actions": lead.actions,
            }
            for lead in leads
        ]
        out_path.write_text(json.dumps(leads_data, indent=2, ensure_ascii=False))
        logger.info("Leads written to '%s'.", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
