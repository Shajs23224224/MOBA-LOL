"""Read-only integration of perception, memory, belief and the evidence council."""
from .agents import Council
from .vision import observe
from .memory import Memory
from .world import World
import torch


class Cognition:
    def __init__(self, path, *, bank=None, capacity=256):
        self.memory = Memory(path,capacity=capacity)
        self.world = World(self.memory)
        self.council = Council()
        self.bank = bank

    def ingest(self, scope, tick, image, context, *, signals=(), ocr=True, neural_agent=None):
        # Validate before persistence; no vision confidence is promoted to a world fact.
        if neural_agent is not None and (self.bank is None or neural_agent not in self.council.by_id):
            raise ValueError('neural diagnostic requires a bank and known agent')
        visual = observe(image,ocr=ocr)
        features = visual['features']
        recalled = self.memory.recall(scope,features)
        deliberation = self.council.deliberate(context)
        neural = None
        if neural_agent is not None:
            output, route = self.council.by_id[neural_agent].neural_diagnostic(self.bank,torch.tensor(features).reshape(1,256))
            neural = {'route':route,'output':output.tolist()[0],'trained':False}
        belief = self.world.update(scope,tick,signals)
        self.memory.put(scope,'sensory',tick,{'regions':visual['regions']},features)
        self.memory.put(scope,'working',tick,{'context':context.model_dump(mode='json'),'recalled_ids':[x['id'] for x in recalled['items']]},features)
        self.memory.put(scope,'episodic',tick,{'context_id':context.id,'recommended':deliberation['recommended']},features)
        self.memory.put(scope,'strategic',tick,{'scores':deliberation['scores'],'outcome':'unobserved'},features)
        self.memory.put(scope,'predictive',tick,belief,features)
        self.memory.put(scope,'metacognitive',tick,{'conflicts':deliberation['conflicts'],'abstentions':deliberation['abstentions']},features)
        return {'visual':visual,'recalled':recalled,'world':belief,'council':deliberation,'neural':neural,'game_control':False}
