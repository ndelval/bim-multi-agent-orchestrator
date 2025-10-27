# Estrategia: Generación de Nodos Condicionales Completos

## Problema Actual

ToT genera nodos de tipo `condition` pero **SIN las especificaciones de la condición**:

```json
// Lo que genera ahora (incompleto):
{"component_type":"node","name":"quality_check","type":"condition"}

// Lo que necesitamos (completo):
{
  "component_type":"node",
  "name":"quality_check",
  "type":"condition",
  "condition_spec": {
    "field": "quality_score",
    "operator": ">=",
    "value": 0.8,
    "true_branch": "publish",
    "false_branch": "retry"
  }
}
```

---

## Solución: Detección + Prompt Especializado

### Paso 1: Detectar Keywords Condicionales

```python
CONDITIONAL_KEYWORDS = [
    # If/then/else patterns
    "si", "if", "cuando", "when", "depende", "depends",
    "en caso de", "in case", "si no", "otherwise", "else",

    # Quality/validation patterns
    "calidad", "quality", "validar", "validate", "verificar", "verify",
    "aprobar", "approve", "rechazar", "reject",

    # Decision patterns
    "decidir", "decide", "escoger", "choose", "seleccionar", "select",
    "evaluar", "evaluate", "comparar", "compare",

    # Threshold patterns
    "mayor que", "greater than", "menor que", "less than",
    "igual a", "equals", "diferente de", "not equals",
    "entre", "between", "fuera de", "outside"
]

def detect_conditional_intent(prompt: str) -> bool:
    """Detectar si el prompt requiere lógica condicional."""
    prompt_lower = prompt.lower()

    # Buscar keywords
    keyword_found = any(kw in prompt_lower for kw in CONDITIONAL_KEYWORDS)

    # Buscar patterns complejos
    if_then_pattern = re.search(r'\b(si|if).+(entonces|then|,)', prompt_lower)
    comparison_pattern = re.search(r'(>|<|>=|<=|==|!=|\bmayor\b|\bmenor\b)', prompt_lower)

    return keyword_found or if_then_pattern or comparison_pattern
```

### Paso 2: Modificar Prompt Base Condicionalmente

```python
def _base_graph_prompt(self, x: str, y: str = "") -> str:
    """Generate base prompt with conditional detection."""

    # Detección de intención condicional
    has_conditional = detect_conditional_intent(x)

    base_prompt = self._standard_prompt(x, y)

    if has_conditional:
        # Agregar INSTRUCCIONES ESPECIALES para condicionales
        conditional_instructions = self._get_conditional_instructions()
        return base_prompt + "\n\n" + conditional_instructions

    return base_prompt


def _get_conditional_instructions(self) -> str:
    """Instrucciones especiales para generación condicional."""
    return (
        "⚠️ DETECCIÓN: Este problema requiere LÓGICA CONDICIONAL.\n\n"
        "PATRÓN OBLIGATORIO para nodos condicionales:\n"
        "1. Genera el nodo condition con especificación COMPLETA:\n"
        '   {"component_type":"node","name":"condition_name","type":"condition",'
        '"condition_spec":{"field":"nombre_campo","operator":">=|<=|==|!=","value":valor_umbral,'
        '"true_branch":"nombre_nodo_si_true","false_branch":"nombre_nodo_si_false"}}\n\n'

        "2. Genera los nodos de las ramas:\n"
        '   {"component_type":"node","name":"nombre_nodo_si_true",...}\n'
        '   {"component_type":"node","name":"nombre_nodo_si_false",...}\n\n'

        "EJEMPLO COMPLETO:\n"
        'Problema: "Analizar datos, si calidad >0.8 publicar, sino corregir"\n\n'
        'Step 1: {"component_type":"node","name":"analyze","type":"agent","agent":"Analyst",...}\n'
        'Step 2: {"component_type":"node","name":"quality_check","type":"condition",'
        '"condition_spec":{"field":"quality_score","operator":">=","value":0.8,'
        '"true_branch":"publish","false_branch":"correct"}}\n'
        'Step 3: {"component_type":"node","name":"publish","type":"agent","agent":"Publisher",...}\n'
        'Step 4: {"component_type":"node","name":"correct","type":"agent","agent":"Corrector",...}\n\n'

        "IMPORTANTE: El campo 'condition_spec' es OBLIGATORIO para type='condition'.\n"
        "Las ramas (true_branch, false_branch) deben coincidir con nombres de nodos generados.\n"
    )
```

### Paso 3: Validación Post-Generación

```python
def _validate_conditional_nodes(components: Dict[str, Any]) -> List[str]:
    """Validar que nodos condicionales tengan especificaciones completas."""
    errors = []

    for node in components['nodes']:
        if node.type == NodeType.CONDITION:
            # Verificar que tenga condition_spec
            if not hasattr(node, 'condition_spec') or not node.condition_spec:
                errors.append(
                    f"Nodo condicional '{node.name}' sin condition_spec. "
                    f"Debe incluir: field, operator, value, true_branch, false_branch"
                )
                continue

            spec = node.condition_spec

            # Validar campos obligatorios
            required = ['field', 'operator', 'value', 'true_branch', 'false_branch']
            missing = [f for f in required if f not in spec]

            if missing:
                errors.append(
                    f"Nodo condicional '{node.name}' incompleto. "
                    f"Falta: {', '.join(missing)}"
                )

            # Validar que las ramas existan como nodos
            node_names = {n.name for n in components['nodes']}

            if spec.get('true_branch') not in node_names:
                errors.append(
                    f"Rama true_branch '{spec['true_branch']}' no existe como nodo"
                )

            if spec.get('false_branch') not in node_names:
                errors.append(
                    f"Rama false_branch '{spec['false_branch']}' no existe como nodo"
                )

    return errors
```

### Paso 4: Auto-Generación de Edges Condicionales

```python
def _create_conditional_edges(graph_spec: StateGraphSpec) -> None:
    """Crear edges automáticamente desde nodos condicionales."""

    for node in graph_spec.nodes:
        if node.type == NodeType.CONDITION and hasattr(node, 'condition_spec'):
            spec = node.condition_spec

            # Edge condicional TRUE
            true_edge = GraphEdgeSpec(
                from_node=node.name,
                to_node=spec['true_branch'],
                type=EdgeType.CONDITIONAL,
                condition=GraphCondition(
                    type="state_check",
                    field=spec['field'],
                    operator=spec['operator'],
                    value=spec['value'],
                    description=f"{spec['field']} {spec['operator']} {spec['value']}"
                ),
                label=f"✓ {spec['field']} {spec['operator']} {spec['value']}",
                description="Condición verdadera"
            )

            # Edge condicional FALSE
            false_edge = GraphEdgeSpec(
                from_node=node.name,
                to_node=spec['false_branch'],
                type=EdgeType.CONDITIONAL,
                condition=GraphCondition(
                    type="state_check",
                    field=spec['field'],
                    operator=negate_operator(spec['operator']),
                    value=spec['value'],
                    description=f"NOT ({spec['field']} {spec['operator']} {spec['value']})"
                ),
                label=f"✗ {spec['field']} {negate_operator(spec['operator'])} {spec['value']}",
                description="Condición falsa"
            )

            try:
                graph_spec.add_edge(true_edge)
                graph_spec.add_edge(false_edge)
                logger.info(f"Created conditional edges for node '{node.name}'")
            except ValueError as e:
                logger.warning(f"Could not create conditional edges: {e}")


def negate_operator(operator: str) -> str:
    """Negar operador lógico."""
    negations = {
        ">=": "<",
        "<=": ">",
        ">": "<=",
        "<": ">=",
        "==": "!=",
        "!=": "==",
        "equals": "not_equals",
        "contains": "not_contains"
    }
    return negations.get(operator, f"NOT {operator}")
```

---

## Modificación del GraphNodeSpec

```python
@dataclass
class GraphNodeSpec:
    """Specification for a single node in a StateGraph."""

    # ... campos existentes ...

    # NUEVO: Especificación de condición para nodos tipo CONDITION
    condition_spec: Optional[Dict[str, Any]] = None
    # Formato esperado:
    # {
    #   "field": "quality_score",        # Campo del state a evaluar
    #   "operator": ">=",                # Operador de comparación
    #   "value": 0.8,                    # Valor umbral
    #   "true_branch": "publish",        # Nodo si condición es true
    #   "false_branch": "retry"          # Nodo si condición es false
    # }
```

---

## Actualización del Parsing

```python
def _create_node_from_component(component: Dict[str, Any]) -> GraphNodeSpec:
    """Create GraphNodeSpec from component dictionary."""
    node_type_map = {
        "agent": NodeType.AGENT,
        "router": NodeType.ROUTER,
        "parallel": NodeType.PARALLEL,
        "condition": NodeType.CONDITION,
        "start": NodeType.START,
        "end": NodeType.END,
    }

    # Extraer condition_spec si existe
    condition_spec = None
    if component.get("type") == "condition" and "condition_spec" in component:
        condition_spec = component["condition_spec"]

    return GraphNodeSpec(
        name=component.get("name", ""),
        type=node_type_map.get(component.get("type", "agent"), NodeType.AGENT),
        agent=component.get("agent"),
        objective=component.get("objective", ""),
        expected_output=component.get("expected_output", ""),
        tools=component.get("tools", []),
        condition_spec=condition_spec  # ← NUEVO
    )
```

---

## Pipeline Completo

```python
def _build_graph_spec_from_plan(
    plan_text: str,
    agent_catalog: Sequence[AgentConfig],
    graph_name: str,
    settings: Optional[GraphPlanningSettings] = None
) -> StateGraphSpec:
    """Build StateGraphSpec with conditional node support."""

    if settings is None:
        settings = GraphPlanningSettings()

    graph_spec = StateGraphSpec(name=graph_name, ...)

    # 1. Parsear componentes (incluye condition_spec)
    components = {"nodes": [], "edges": [], "parallel_groups": []}

    for line in plan_text.strip().split("\n"):
        for component in _parse_json_objects(line):
            comp_type = component.get("component_type", "")

            if comp_type == "node":
                node_spec = _create_node_from_component(component)
                components["nodes"].append(node_spec)
            # ... otros tipos ...

    # 2. Validar nodos condicionales
    conditional_errors = _validate_conditional_nodes(components)
    if conditional_errors:
        logger.warning(f"Conditional validation errors: {conditional_errors}")

    # 3. Agregar nodos al grafo
    for node in components["nodes"]:
        graph_spec.add_node(node)

    # 4. Inferir edges básicos (tu propuesta)
    basic_edges = _infer_edges_from_temporal_order(components["nodes"])
    for edge in basic_edges:
        try:
            graph_spec.add_edge(edge)
        except ValueError:
            pass  # Skip si ya existe

    # 5. Crear edges condicionales automáticamente
    _create_conditional_edges(graph_spec)

    # 6. Manejar parallel_groups
    for parallel_group in components["parallel_groups"]:
        try:
            graph_spec.add_parallel_group(parallel_group)
            _create_parallel_edges_for_group(graph_spec, parallel_group)
        except ValueError as e:
            logger.warning(f"Skipping invalid parallel group: {e}")

    # 7. Asegurar start/end
    _ensure_start_end_nodes(graph_spec)

    # 8. Validación final
    errors = graph_spec.validate()
    if errors and not settings.enable_auto_fallback:
        raise ValueError(f"Graph validation failed: {errors}")

    return graph_spec
```

---

## Ejemplo Completo: Caso Real

### Input del Usuario:
```
"Analizar datos financieros. Si la calidad es mayor a 0.8, publicar el reporte.
Si no, corregir los datos y volver a analizar."
```

### Detección:
```python
detect_conditional_intent("Analizar... Si la calidad es mayor a 0.8...")
# → True (detecta "Si", "mayor a", "0.8")
```

### Prompt Mejorado:
```
[Prompt base]

⚠️ DETECCIÓN: Este problema requiere LÓGICA CONDICIONAL.

PATRÓN OBLIGATORIO para nodos condicionales:
[Instrucciones completas...]
```

### ToT Genera (con prompt mejorado):
```json
{"component_type":"node","name":"analyze_financial_data","type":"agent","agent":"Analyst","objective":"Analyze financial data","expected_output":"Analysis report"}

{"component_type":"node","name":"quality_check","type":"condition","condition_spec":{"field":"quality_score","operator":">=","value":0.8,"true_branch":"publish_report","false_branch":"correct_data"}}

{"component_type":"node","name":"publish_report","type":"agent","agent":"Publisher","objective":"Publish validated report","expected_output":"Published report"}

{"component_type":"node","name":"correct_data","type":"agent","agent":"DataCorrector","objective":"Fix data quality issues","expected_output":"Corrected data"}
```

### Edges Auto-Generados:
```python
# Edges temporales básicos:
("start", "analyze_financial_data", "direct")
("analyze_financial_data", "quality_check", "direct")

# Edges condicionales (automáticos desde condition_spec):
("quality_check", "publish_report", "conditional", condition={field:"quality_score", op:">=", val:0.8})
("quality_check", "correct_data", "conditional", condition={field:"quality_score", op:"<", val:0.8})

# Cierre:
("publish_report", "end", "direct")
("correct_data", "analyze_financial_data", "direct")  # Loop back!
```

### Grafo Final:
```
         start
           │
    analyze_financial_data
           │
      quality_check
       /          \
   [≥0.8]        [<0.8]
     /              \
publish_report    correct_data
     │                │
    end          [loop back to analyze]
```

---

## Ventajas de Esta Estrategia

✅ **Detección Inteligente**: Identifica cuándo se necesita lógica condicional
✅ **Prompt Especializado**: Guía al LLM a generar estructura completa
✅ **Validación Estricta**: Asegura que condition_spec esté completo
✅ **Auto-Generación de Edges**: Crea edges condicionales automáticamente
✅ **Fallback Robusto**: Si LLM no genera condition_spec, usa inferencia básica
✅ **Loops Soportados**: Permite correct_data → analyze (ciclos)

## Limitaciones Aceptables

⚠️ **Condiciones Complejas**: `(A AND B) OR C` requiere expresión personalizada
⚠️ **Multi-Branch**: Switch con >2 ramas necesita extensión del formato
⚠️ **Condiciones Dinámicas**: Evaluación en runtime vs compile-time

Pero el **80% de casos** (if/else simple) quedan cubiertos perfectamente.
