Configuración Actual del Árbol

# orchestrator/planning/tot_graph_planner.py líneas 42-44

n_generate_sample: int = 3 # Genera 3 ramas en cada step
n_evaluate_sample: int = 2 # Evalúa cada rama 2 veces para sacar score promedio
n_select_sample: int = 2 # Selecciona las 2 mejores ramas para continuar
max_steps: int = 5 # Profundidad máxima del árbol

Visualización del Árbol que se Genera

Step 0: INICIO
═══════════════════════════════════════════════════════════
ys = [''] ← Comienza con string vacío

Llamada a GPT: "Genera el primer componente del grafo"
├─ n=3 (genera 3 opciones diferentes)

GPT devuelve 3 alternativas:
├─ Opción A: {"component_type":"node","name":"research_task",...}
├─ Opción B: {"component_type":"node","name":"gather_data",...}
└─ Opción C: {"component_type":"node","name":"analyze_info",...}

new_ys = [A, B, C] ← 3 candidatos

Step 0: EVALUACIÓN
───────────────────────────────────────────────────────────
Llamada a GPT: "Evalúa qué tan bueno es cada candidato"
├─ n=2 por cada candidato (para promediar scores)

Evaluación de A: GPT dice score=8.5, 9.0 → promedio = 8.75
Evaluación de B: GPT dice score=7.0, 8.0 → promedio = 7.5
Evaluación de C: GPT dice score=9.0, 8.5 → promedio = 8.75

values = [8.75, 7.5, 8.75]

Step 0: SELECCIÓN (greedy)
───────────────────────────────────────────────────────────
n_select_sample = 2 ← Selecciona las 2 mejores

Ranking:

1. Opción A: 8.75 ✓ SELECCIONADA
2. Opción C: 8.75 ✓ SELECCIONADA
3. Opción B: 7.5 ✗ DESCARTADA

ys = [A, C] ← Solo estas 2 pasan al siguiente step

═══════════════════════════════════════════════════════════
Step 1: EXPANSIÓN desde 2 ramas
═══════════════════════════════════════════════════════════
Ahora tenemos ys = [A, C]

Para cada rama, genera 3 continuaciones:

Rama A (research_task):
├─ Llamada a GPT con contexto A
│ └─ Genera 3 opciones: A1, A2, A3
│ A1 = A + {"component_type":"node","name":"analyze",...}
│ A2 = A + {"component_type":"parallel_group",...}
│ A3 = A + {"component_type":"edge","from":"start",...}

Rama C (analyze_info):  
 └─ Llamada a GPT con contexto C
└─ Genera 3 opciones: C1, C2, C3
C1 = C + {"component_type":"node","name":"review",...}
C2 = C + {"component_type":"node","name":"validate",...}
C3 = C + {"component_type":"parallel_group",...}

new_ys = [A1, A2, A3, C1, C2, C3] ← 6 candidatos totales

Step 1: EVALUACIÓN
───────────────────────────────────────────────────────────
Evalúa los 6 candidatos (2 veces cada uno):

A1: score = 8.0
A2: score = 9.5 ← ¡Mejor! Tiene parallel_group
A3: score = 7.0
C1: score = 6.5
C2: score = 7.5
C3: score = 8.5

values = [8.0, 9.5, 7.0, 6.5, 7.5, 8.5]

Step 1: SELECCIÓN
───────────────────────────────────────────────────────────
Selecciona las 2 mejores:

1. A2: 9.5 ✓
2. C3: 8.5 ✓

ys = [A2, C3]

═══════════════════════════════════════════════════════════
Step 2, 3, 4: CONTINUACIÓN
═══════════════════════════════════════════════════════════
El proceso se repite...
Pero aquí está el PROBLEMA:

EL PROBLEMA: Búsqueda con Poda Agresiva

CONFIGURACIÓN ACTUAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Genera: 3 ramas por step (n_generate_sample=3)
Evalúa: 2 veces por rama (n_evaluate_sample=2)
Poda: Solo 2 sobreviven (n_select_sample=2)
Depth: 5 steps máximo (max_steps=5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ÁRBOL REAL GENERADO:

                      ['']
                       │
          ┌────────────┼────────────┐
          │            │            │
         [A]          [B]          [C]
       (8.75)        (7.5)        (8.75)
          │            ✗            │
          │         PODADO          │
          │                         │
      ┌───┼───┐               ┌────┼────┐
     [A1][A2][A3]            [C1][C2][C3]
     (8) (9.5)(7)            (6.5)(7.5)(8.5)
      ✗   │   ✗              ✗   ✗   │
          │                           │
          │                           │
        [A2]                        [C3]
     (continúa)                  (continúa)
          │                           │
          └───────────┬───────────────┘
                      │
                Steps 3, 4, 5
                (más expansión)

RESULTADO FINAL DE TU CASO

// Lo que ToT generó en 5 steps:
best_plan =
'{"component_type":"node","name":"gather_market_data","type":"agent","agent":"Researcher",...}\n' +
'{"component_type":"node","name":"analyze_market_data","type":"agent","agent":"Analyst",...}\n' +
'{"component_type":"parallel_group","name":"market_analysis_workflow","parallel_nodes":[...]}\n'

¿Por Qué No Hay Edges?

Teoría del Árbol de Decisión:

Step 0: Genera PRIMER componente
├─ Opción: node (Researcher) ← Alta prioridad (conceptualmente simple)
├─ Opción: node (Analyst) ← Alta prioridad
└─ Opción: edge (start→research) ← Baja prioridad (abstracto, menos obvio)

GPT evalúa: "Los nodes son más importantes que edges"
→ Selecciona los 2 mejores: ambos son NODES

Step 1: Genera SEGUNDO componente (desde nodes)
├─ Opción: otro node ← Sigue siendo obvio
├─ Opción: parallel_group ← ¡Innovador! GPT lo valora alto
└─ Opción: edge ← Menos "interesante"

GPT evalúa: "parallel_group es sofisticado y eficiente"
→ Selecciona: node + parallel_group

Steps 2-5: Genera TERCER y más componentes
├─ Ya tiene 2 nodes + 1 parallel_group
├─ GPT piensa: "El grafo ya está estructurado"
├─ Opciones de edges tienen scores bajos
└─ Las 2 mejores ramas NO incluyen edges

RESULTADO: 3 JSON objects, 0 edges

Llamadas Reales a GPT en Tu Caso

# TOTAL de llamadas a GPT en 5 steps:

# (asumiendo siempre 2 ramas sobreviven)

Step 0: - Generación: 1 llamada × n=3 opciones = 3 outputs - Evaluación: 3 candidatos × n=2 evaluaciones = 6 llamadas
Total: 7 llamadas

Step 1: - Generación: 2 ramas × n=3 opciones = 6 outputs (2 llamadas) - Evaluación: 6 candidatos × n=2 evaluaciones = 12 llamadas
Total: 14 llamadas

Step 2: - Generación: 2 ramas × n=3 opciones = 6 outputs (2 llamadas) - Evaluación: 6 candidatos × n=2 evaluaciones = 12 llamadas
Total: 14 llamadas

Steps 3, 4: Similar...

TOTAL APROXIMADO: 50-70 llamadas a GPT
TOKENS CONSUMIDOS: ~50K-100K tokens (depende de complejidad)
TIEMPO: 30-60 segundos

Visualización del Árbol Completo

                                      START: ''
                                         │
                      ┌──────────────────┼──────────────────┐
                      │                  │                  │
                    Node A             Node B           Node C
                   (score=8.75)      (score=7.5)     (score=8.75)
                      │                  ✗                  │
                      │              PRUNED                 │
                      │                                     │
           ┌──────────┼──────────┐           ┌─────────────┼─────────────┐
           │          │          │           │             │             │
        +Node     +ParaGrp    +Edge      +Node         +Node       +ParaGrp
        (8.0)       (9.5)      (7.0)     (6.5)         (7.5)         (8.5)
           ✗          │          ✗         ✗             ✗             │
                      │                                                 │
                SELECTED                                          SELECTED
                      │                                                 │
                      └────────────────┬───────────────────────────────┘
                                       │
                                Step 2 (continúa)
                                       │
                          ┌────────────┼────────────┐
                          │            │            │
                      +Edge        +Node        +Quality
                      (6.0)        (7.5)         (8.0)
                        ✗            ✗             │
                                                   │
                                              SELECTED
                                                   │
                                          Steps 3-5 (expandir)
                                                   │
                                             FINAL OUTPUT:
                                       3 JSON lines (2 nodes + 1 para_group)
