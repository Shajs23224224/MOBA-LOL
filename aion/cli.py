import argparse,json,logging,time
from .config import load_settings
from .catalog import architecture
from .models import atomic_json,ModelRegistry
from .resources import hardware,ResourceBudget
from .runtime import Engine
from .storage import Store

def main(argv=None):
    p=argparse.ArgumentParser(description='AION phase 1 infrastructure')
    p.add_argument('--config'); sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor'); a=sub.add_parser('architecture'); a.add_argument('--output')
    run=sub.add_parser('run'); run.add_argument('--seconds',type=float,default=2)
    sub.add_parser('status'); sub.add_parser('models'); sub.add_parser('neural-probe')
    args=p.parse_args(argv); logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    settings=load_settings(args.config)
    if args.command=='doctor': result=hardware()
    elif args.command=='architecture':
        result=architecture()
        if args.output:
            atomic_json(args.output,result); result={k:v for k,v in result.items() if k not in ('networks','agents')}
    elif args.command=='models': result=ModelRegistry(settings.data_dir/'models').audit()
    elif args.command=='status': result=Store(settings.data_dir/'aion.sqlite3').summary()
    elif args.command=='neural-probe':
        from .probe import neural_probe
        result=neural_probe(ResourceBudget(settings).choose_device())
    elif args.command=='run':
        if not 0<args.seconds<=3600: p.error('--seconds must be > 0 and <= 3600')
        engine=Engine(settings); engine.start()
        try: time.sleep(args.seconds)
        except KeyboardInterrupt: pass
        finally: engine.stop()
        result=engine.snapshot()
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 1 if isinstance(result,dict) and (result.get('passed') is False or result.get('state')=='failed') else 0

if __name__=='__main__': raise SystemExit(main())
