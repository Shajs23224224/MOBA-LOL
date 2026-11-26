"""Separate neural diagnostics CLI; leaves the original phase 1 entry point intact."""
import argparse
import json
import torch
from .neural import ExpertBank, INPUT
from .config import load_settings
from .models import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description='AION phase 2 neural diagnostics')
    parser.add_argument('--config')
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    settings = load_settings(args.config)
    bank = ExpertBank(registry=ModelRegistry(settings.data_dir/'models'))
    report = {'audit_before': bank.audit()}
    if args.infer:
        output, route = bank.infer(torch.ones(1, INPUT))
        report.update(route=route, finite=bool(torch.isfinite(output).all()), shape=list(output.shape), audit_after=bank.audit())
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
