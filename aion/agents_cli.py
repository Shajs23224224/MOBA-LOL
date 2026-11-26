"""Run a deliberation on user-supplied research evidence; no actions are executed."""
import argparse
import json
from pathlib import Path
from .agents import Council, Context


def main():
    p = argparse.ArgumentParser(description='AION evidence council diagnostics')
    p.add_argument('context', type=Path)
    p.add_argument('--neural-agent')
    a = p.parse_args()
    council = Council()
    context = Context.model_validate_json(a.context.read_text(encoding='utf-8'))
    report = council.deliberate(context)
    if a.neural_agent:
        import torch
        from .neural import ExpertBank, INPUT
        agent = council.by_id[a.neural_agent]
        output, route = agent.neural_diagnostic(ExpertBank(),torch.ones(1, INPUT))
        report['neural_diagnostic'] = {'agent':agent.id,'route':route,'finite':bool(torch.isfinite(output).all()),'shape':list(output.shape),'trained':False}
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
