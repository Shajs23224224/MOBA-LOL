"""Unified offline research pipeline; no actuator or executable policy."""
from math import tanh
import torch
from .catalog import SUBSYSTEMS, networks
from .agents import GROUPS
from .moba import review
from .memory import KINDS
from .fusion import Fusion
from .learning import ActorCritic
from .contracts import ActionIntent


def summaries(report, council):
    memory = report['moba']['memory_by_kind']
    common = []
    for kind in KINDS:
        items = memory[kind]['items']
        common.extend((float(bool(items)),max((x['similarity'] for x in items),default=0.)))
    beliefs = list(report['cognition']['world']['beliefs'].values())
    common.extend((float(bool(beliefs)),len(beliefs)/4096,
                   sum(b['p'] for b in beliefs)/max(1,len(beliefs)),
                   sum(b['observed_this_tick'] for b in beliefs)/max(1,len(beliefs))))
    departments = tuple(dict.fromkeys(a.department for a in council.agents))
    ranking = report['planning']['ranked']
    estimates = report['moba']['estimates']
    plan = [len(ranking)/16,tanh(ranking[0]['value']) if ranking else 0.,
            tanh(ranking[-1]['value']) if ranking else 0.,float(report['planning']['requires_review']),
            len(report['cognition']['council']['conflicts'])/16,
            max((e['risk'] for e in estimates),default=0.),
            max((e['probability'] for e in estimates),default=0.),
            float(report['planning']['corroborated_label'] is not None)]
    rows=[]
    for group in SUBSYSTEMS:
        ids={a.id for a in council.agents if group in GROUPS[departments.index(a.department)]}
        proposals=[p for p in report['cognition']['council']['proposals'] if p['agent_id'] in ids]
        role=[len({p['agent_id'] for p in proposals})/max(1,len(ids)),
              sum(p['utility'] for p in proposals)/max(1,len(proposals)),
              sum(p['confidence'] for p in proposals)/max(1,len(proposals)),
              max((p['risk'] for p in proposals),default=0.)]
        rows.append(common+role+plan)
    return torch.tensor(rows,dtype=torch.float32)


class Unified:
    def __init__(self,cognition,*,fusion=None,learner=None):
        self.cognition=cognition
        self.fusion=(fusion if fusion is not None else Fusion()).eval()
        self.learner=(learner if learner is not None else ActorCritic()).eval()

    def audit(self):
        f=self.fusion.audit()
        a=sum(p.numel() for p in self.learner.parameters())
        bank=self.cognition.bank.audit() if self.cognition.bank is not None else None
        return {'fusion':f,'actor_critic_parameters':a,'bank':bank,
                'catalog_networks':858,'logical_total_parameters':bank['logical_total_parameters']+f['parameters']+a if bank else None,
                'resident_parameter_bytes':(bank['resident_weight_bytes'] if bank else 0)+f['resident_bytes']+sum(p.numel()*p.element_size() for p in self.learner.parameters())}

    @torch.inference_mode()
    def run(self,scope,tick,image,snapshot,*,evidence=(),signals=(),subsystems=(),ocr=True):
        subsystems=tuple(subsystems)
        if len(subsystems)>2 or len(set(subsystems))!=len(subsystems) or not set(subsystems)<=set(SUBSYSTEMS):
            raise ValueError('select zero to two unique subsystems')
        if subsystems and self.cognition.bank is None:
            raise ValueError('expert bank required')
        report=review(self.cognition,scope,tick,image,snapshot,evidence=evidence,signals=signals,ocr=ocr)
        visual=torch.tensor(report['cognition']['visual']['features']).reshape(1,256)
        experts,routes={},{}
        for group in subsystems:
            ids=[n.id for n in networks() if n.kind=='main' and n.subsystem==group]
            experts[group],routes[group]=self.cognition.bank.infer(visual,candidates=ids)
        actor,value,prediction=self.learner(visual)
        result=self.fusion(visual,summaries(report,self.cognition.council),experts,
                           torch.cat((actor.softmax(-1),value.reshape(1,1)),dim=1))
        if any(not torch.isfinite(t).all() for t in (*result['representations'].values(),*result['estimates'].values(),result['strategic_logits'],result['policy_logits'],prediction)):
            raise ValueError('nonfinite unified output')
        options=[r['option'] for r in report['moba']['estimates']]
        integrated={'routes':routes,'subsystems':{g:{'vector':result['representations'][g][0].tolist(),
                     'uncalibrated_estimates':dict(zip(('value','risk','confidence','uncertainty'),result['estimates'][g][0].tolist())),
                     'expert_available':g in experts} for g in SUBSYSTEMS},
                    'strategic_diagnostic':dict(zip(options,result['strategic_logits'][0].tolist())),
                    'policy_diagnostic':result['policy_logits'][0].softmax(-1).tolist(),
                    'actor_diagnostic':actor[0].softmax(-1).tolist(),'actor_value':float(value[0]),
                    'next_feature_prediction':prediction[0].tolist(),'calibrated':False,
                    'research_candidate':report['planning']['research_candidate'],
                    'corroborated_label':report['planning']['corroborated_label'],
                    'requires_review':True,'intent':ActionIntent(observation_id=f'moba-{tick}').model_dump(),
                    'game_control':False}
        self.cognition.memory.put(scope,'metacognitive',tick,{'fusion_review':{'routes':routes,'requires_review':True}},visual[0].tolist())
        return {'report':report,'integration':integrated,'audit':self.audit()}
