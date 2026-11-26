"""Stable architecture identifiers. Catalog entries are requirements, not instantiated models."""
from dataclasses import dataclass, asdict

SUBSYSTEMS = ('percepcion_visual','reconocimiento_entidades','cartografia','estado_juego','memoria','logica_causal','prediccion','planificacion','combate','economia','compras','navegacion','cooperacion','aprendizaje','metacognicion','coordinacion')
ROLES = {
'percepcion': 'campeones|enemigos|aliados|subditos|monstruos_neutrales|estructuras|habilidades_visuales|proyectiles|lectura_interfaz|seguimiento_temporal',
'cartografia': 'minimapa|rutas|terreno|obstaculos|zonas_peligro|arbustos|vision|niebla_guerra|exploracion|control_territorial',
'razonamiento': 'deduccion|induccion|abduccion|causalidad|probabilidad|hipotesis|contradicciones|resolucion_problemas|alternativas|contrafactuales',
'estrategia': 'juego_temprano|juego_medio|juego_tardio|condiciones_victoria|presion_territorial|objetivos|ventajas|recuperacion|largo_plazo|adaptacion',
'combate': 'duelos|intercambio_dano|combate_multiple|posicionamiento|seleccion_objetivo|esquiva|secuencias_habilidades|retirada|persecucion|supervivencia',
'economia': 'oro|adquisicion_recursos|eficiencia|experiencia_niveles|mana|energia|salud|recuperacion|coste_oportunidad|recompensas',
'compras': 'ofensivos|defensivos|situacionales|componentes|eficiencia_compra|sinergias|composiciones|resistencia_fisica|resistencia_magica|configuracion',
'prediccion': 'movimientos|ataques|habilidades|dano|rutas|amenazas|enfrentamientos|adversario|evolucion|victoria',
'posiciones': 'superior|central|inferior|jungla|soporte|rotaciones|presion_lateral|emboscadas|contraemboscadas|cambios_posicion',
'coordinacion_colectiva': 'proteccion|iniciacion|coordinacion_grupal|composicion|funciones|apoyo|prioridades|sincronizacion|cooperacion|sacrificios_estrategicos',
'objetivos': 'oleadas|subditos|presion_estructuras|monstruos|prioridades|tiempo|preparacion|defensa|avance|finalizacion',
'aprendizaje': 'refuerzo|autosupervision|experiencia|temporalidad|generalizacion|transferencia|novedades|exploracion|consolidacion|prevencion_olvido',
'memoria': 'inmediata|trabajo|episodica|semantica|procedimental|recuperacion|asociacion|compresion|relevancia|persistencia',
'metacognicion': 'errores|autocritica|rendimiento|incertidumbre|revision|validacion|inconsistencias|optimizacion|supervision|evaluacion',
'decisiones': 'teoria_juegos|riesgo|utilidad|alternativas|multiobjetivo|jerarquia|conflictos|prioridad|consecuencias|politicas',
'ejecutivo': 'percepcion|memoria|razonamiento|estrategia|combate|economia|prediccion|aprendizaje|supervision_metacognitiva|central'}
@dataclass(frozen=True)
class NetworkSpec:
    id: str
    kind: str
    subsystem: str
    hidden_layers: int | None
    hidden_width: int | None
    purpose: str
    status: str = 'planned'
@dataclass(frozen=True)
class AgentSpec:
    id: str
    department: str
    role: str
    status: str = 'planned'

def networks():
    entries = []
    for i, group in enumerate(SUBSYSTEMS):
        for j in range(52 if i < 8 else 51):
            entries.append(NetworkSpec(f'N{len(entries)+1:03}', 'main', group, 16, 2048, f'{group}: experto {j+1}'))
    for i, group in enumerate(SUBSYSTEMS):
        for j, purpose in enumerate(('integration','confidence')):
            entries.append(NetworkSpec(f'I{i*2+j+1:02}', 'intermediate', group, None, None, purpose))
    entries.extend([NetworkSpec('F01','final','coordinacion',None,None,'strategic_decision'), NetworkSpec('F02','final','coordinacion',None,None,'action_policy')])
    return tuple(entries)

def agents():
    entries = []
    for group, roles in ROLES.items():
        for role in roles.split('|'):
            entries.append(AgentSpec(f'A{len(entries)+1:03}',group,role))
    return tuple(entries)

def architecture():
    ns, ag = networks(), agents()
    return {'main_count':824,'intermediate_count':32,'final_count':2,'network_count':len(ns),'agent_count':len(ag), 'main_hidden_units':824*16*2048,'main_hidden_layers':824*16,'dense_hidden_to_hidden_weights':824*15*2048**2,'materialized_main_networks':0,'functional_expert_agents':0,'networks':[asdict(x) for x in ns],'agents':[asdict(x) for x in ag]}
