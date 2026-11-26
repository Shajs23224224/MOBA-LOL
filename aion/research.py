"""Explicit research planning attached to the Phase 5 read-only cognitive pipeline."""
from .planning import parallel_roots


def analyze(cognition, scope, tick, image, context, model, start, *, signals=(), neural_agent=None, ocr=True, depth=5, iterations=128):
    report=cognition.ingest(scope,tick,image,context,signals=signals,neural_agent=neural_agent,ocr=ocr)
    ranked=parallel_roots(model,start,depth=depth,iterations=iterations)
    # Only compare labels; never manufacture observed evidence from simulated outcomes.
    proposed=ranked[0]['option'] if ranked else None
    agreed=proposed if proposed is not None and proposed==report['council']['recommended'] else None
    result={'ranked':ranked,'research_candidate':proposed,'corroborated_label':agreed,
            'requires_review':agreed is None,'belief_snapshot':report['world'],
            'recalled_ids':[x['id'] for x in report['recalled']['items']],
            'game_control':False}
    cognition.memory.put(scope,'strategic',tick,{'research_plan':result,'outcome':'unobserved'},report['visual']['features'])
    return {'cognition':report,'planning':result}
