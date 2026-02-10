# Datasets CSV - Guía Completa

## ¿Qué son estos archivos CSV?

Estos 3 archivos CSV contienen **ejemplos de entrenamiento** para enseñar al LLM (GPT-4) cómo procesar queries BIM en lenguaje natural. Funcionan como "profesores" que muestran al AI ejemplos de cómo debe comportarse.

Este enfoque se llama **few-shot learning**: el LLM aprende de los ejemplos incluidos en el prompt y aplica ese conocimiento a nuevas queries.

---

## Los 3 Archivos CSV

### 1. `intent_examples.csv` - Clasificación de Intenciones

**Propósito**: Enseñar al LLM a clasificar queries en categorías.

**Columnas**:
- `query`: La pregunta del usuario en lenguaje natural
- `intent`: La categoría de la pregunta
- `is_bim`: Si es una pregunta relacionada con BIM o no

**Categorías de intenciones**:
- `location`: Elementos en ubicaciones específicas
  - Ejemplo: "¿Qué muros hay en el Nivel 1?"
- `material`: Elementos de cierto material
  - Ejemplo: "Muéstrame todas las columnas de hormigón"
- `quantity`: Contar elementos
  - Ejemplo: "¿Cuántas puertas hay?"
- `property`: Consultas sobre propiedades específicas
  - Ejemplo: "¿Cuál es el volumen de los muros?"
- `spatial`: Relaciones espaciales entre elementos
  - Ejemplo: "¿Qué vigas están conectadas a esta columna?"
- `other`: Consultas no relacionadas con BIM
  - Ejemplo: "¿Qué tiempo hace hoy?"

**Contenido actual**: 16 ejemplos

**Ejemplo de uso en el código**:
```python
# Block 1 carga estos ejemplos y los incluye en el prompt
intent_classifier = IntentClassifier(llm_client, dataset_path)
result = intent_classifier.classify("¿Qué muros hay en el Nivel 1?")
# → result.intent = "location"
```

---

### 2. `parameter_examples.csv` - Extracción de Parámetros

**Propósito**: Enseñar al LLM a extraer **qué filtrar** y **qué propiedades retornar** de una query.

**Columnas**:
- `query`: La pregunta del usuario
- `intent`: Categoría de la query (del paso anterior)
- `filter_para`: JSON con filtros a aplicar
- `proj_para`: JSON con propiedades a proyectar/retornar

**Ejemplo de fila**:
```csv
query,intent,filter_para,proj_para
"What walls are on Level 1?","location","[{""parameter_name"":""Category"",""value"":""Walls"",""operator"":""EQUALS""},{""parameter_name"":""Level"",""value"":""Level 1"",""operator"":""EQUALS""}]","[""Name"",""Volume"",""Area"",""Material""]"
```

Esto enseña al LLM que:
- **Filtros**: Buscar donde Category = "Walls" AND Level = "Level 1"
- **Proyecciones**: Retornar Name, Volume, Area, Material

**Parámetros comunes**:
- **Categorías**: Walls, Doors, Windows, Columns, Beams, Floors
- **Propiedades**: Category, Level, Material, Volume, Area, Height, Width, Fire Rating

**Contenido actual**: 12 ejemplos

**Ejemplo de uso**:
```python
# Block 2 extrae parámetros basándose en estos ejemplos
param_extractor = ParamExtractor(llm_client, dataset_path)
result = param_extractor.extract(query, intent)
# → result.filter_para = [FilterParameter(...)]
# → result.proj_para = ["Name", "Volume", "Area"]
```

---

### 3. `value_resolution_examples.csv` - Normalización de Valores

**Propósito**: Enseñar al LLM a normalizar lenguaje coloquial a valores estándar BIM.

**Columnas**:
- `query`: La pregunta original
- `filter_para`: Parámetros con valores sin normalizar
- `normalized_values`: Valores normalizados en formato JSON

**Normalizaciones comunes**:

| Usuario dice | Sistema normaliza |
|--------------|-------------------|
| "first floor" | "Level 1" |
| "second floor" | "Level 2" |
| "ground floor" | "Level 1" |
| "concrete" | "Concrete" |
| "steel" | "Steel" |
| "2hr fire rated" | "2 Hour" |
| "lvl 2" | "Level 2" |

**Ejemplo de fila**:
```csv
query,filter_para,normalized_values
"Show walls on first floor","[{""parameter_name"":""Level"",""value"":""first floor""}]","[{""parameter_name"":""Level"",""original_value"":""first floor"",""normalized_value"":""Level 1"",""is_standardized"":true}]"
```

**Contenido actual**: 11 ejemplos

**Ejemplo de uso**:
```python
# Block 3 normaliza valores basándose en estos ejemplos
value_resolver = ValueResolver(llm_client, dataset_path)
result = value_resolver.resolve(params, query)
# Input: "first floor" → Output: "Level 1"
```

---

## ¿Por qué son necesarios?

**Sin estos CSVs, el sistema NO funciona** porque:

1. **El LLM necesita ejemplos**: GPT-4 es muy capaz, pero necesita ver ejemplos específicos del dominio BIM para dar resultados precisos.

2. **Estandarización**: BIM usa nomenclatura específica (Revit, IFC) que el usuario puede desconocer. Los ejemplos enseñan la correspondencia entre lenguaje natural y términos BIM.

3. **Consistencia**: Garantizan que las respuestas sigan un formato parseable por el código Python.

---

## Formato JSON dentro de CSV

Los campos que contienen JSON usan **comillas escapadas**:

```csv
"query","filter_para"
"Show walls","[{""parameter_name"":""Category"",""value"":""Walls""}]"
```

Las comillas dobles dentro del JSON se duplican (`""`) según el estándar CSV.

Python lee esto correctamente:
```python
import pandas as pd
df = pd.read_csv('intent_examples.csv')
# Las comillas dobles se convierten a simples automáticamente
print(df['filter_para'][0])
# → [{"parameter_name":"Category","value":"Walls"}]
```

---

## Cómo Agregar Más Ejemplos

### Opción 1: Editar los CSVs directamente

Abre los archivos CSV y agrega filas siguiendo el formato existente.

**Importante**: 
- Usa el formato JSON correcto con comillas escapadas
- Sigue la nomenclatura BIM estándar (Title Case)
- Asegúrate de que los ejemplos sean consistentes entre archivos

### Opción 2: Modificar el script generador

Edita `create_sample_datasets.py` y vuelve a ejecutarlo:

```python
# Agregar ejemplo a intent_examples.csv
{
    "query": "¿Qué columnas son de acero?",
    "intent": "material",
    "is_bim": "true"
}
```

Luego ejecuta:
```bash
python3 agents/bim_ir/datasets/create_sample_datasets.py
```

---

## Mejores Prácticas

### Diversidad de ejemplos
- Cubre todos los casos comunes: muros, columnas, puertas, ventanas, vigas
- Incluye variaciones en lenguaje: "first floor", "nivel 1", "planta baja"
- Agrega ejemplos negativos (is_bim=false) para que el LLM rechace queries irrelevantes

### Consistencia de nomenclatura
- **Categorías**: Walls, Doors, Windows (plural, Title Case)
- **Niveles**: "Level 1", "Level 2" (no "lvl 1", "1st floor")
- **Materiales**: "Concrete", "Steel", "Brick" (Title Case)
- **Propiedades**: Volume, Area, Height, Width, "Fire Rating" (Title Case)

### Calidad sobre cantidad
- 10-15 ejemplos bien diseñados > 50 ejemplos mediocres
- Asegúrate de que el JSON sea válido
- Verifica que los parámetros existan realmente en modelos BIM

---

## Troubleshooting

### Error: "Could not parse JSON"
- Verifica que las comillas estén escapadas correctamente (`""`)
- Usa un validador JSON online para verificar el formato

### Error: "No examples found"
- Verifica que los archivos CSV estén en `agents/bim_ir/datasets/`
- Verifica que tengan al menos 1 ejemplo cada uno

### LLM da resultados inconsistentes
- Agrega más ejemplos para el caso problemático
- Asegúrate de que los ejemplos sean claros y no ambiguos
- Verifica que la nomenclatura sea consistente

---

## Estadísticas Actuales

| Archivo | Ejemplos | Propósito |
|---------|----------|-----------|
| intent_examples.csv | 16 | Clasificar intención de la query |
| parameter_examples.csv | 12 | Extraer filtros y proyecciones |
| value_resolution_examples.csv | 11 | Normalizar valores coloquiales |
| **TOTAL** | **39** | Few-shot learning completo |

---

## Next Steps

1. **Probar el sistema** con los ejemplos actuales:
   ```bash
   python3 agents/bim_ir/tests/test_nlu_integration.py
   ```

2. **Expandir datasets** si necesitas mejor accuracy:
   - Agrega ejemplos para queries específicas de tu dominio
   - Incluye variaciones de lenguaje de tus usuarios
   - Cubre edge cases que encuentres en producción

3. **Medir accuracy** y agregar ejemplos donde fallan las predictions

---

## Referencias

- **BIM-GPT Paper**: Describe el enfoque de few-shot learning para BIM
- **Blocks 1-3 Documentation**: Ver `claudedocs/PHASE3_BLOCK*_COMPLETE.md`
- **OpenAI Few-Shot Learning**: https://platform.openai.com/docs/guides/prompt-engineering

---

**Creado**: 2025-11-01  
**Ejemplos totales**: 39  
**Última actualización**: Inicial release
