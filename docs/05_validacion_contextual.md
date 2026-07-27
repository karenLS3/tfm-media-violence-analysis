# Validación de la clasificación contextual

## Objetivo

La validación tiene como objetivo estimar la fiabilidad de la capa de clasificación contextual y justificar qué grupos pueden utilizarse en el corpus principal, cuáles deben reservarse para análisis de sensibilidad y cuáles deben mantenerse fuera de los resultados principales.

La validación no pretende revisar manualmente todos los artículos. Se aplica sobre una muestra estratificada que incluye casos con distintos niveles de evidencia y grupos de control.

## Relación con la recuperación inicial

La validación evalúa la clasificación contextual añadida sobre `retrieval_bucket`.

Ambas capas tienen funciones diferentes:

1. `retrieval_bucket` representa la recuperación inicial mediante reglas amplias.
2. La clasificación contextual estima el grado de evidencia de que el artículo describe un caso relacionado con violencia contra las mujeres y la forma en que el medio reconoce el componente de género.

La validación no sustituye la recuperación ni modifica sus categorías originales.

## Scripts principales

La muestra se construye mediante:

```powershell
python -m src.analysis.build_contextual_validation_sample
```

La evaluación de las etiquetas manuales se realiza mediante:

```powershell
python -m src.analysis.evaluate_contextual_validation
```

Los nombres exactos de los argumentos y archivos de entrada pueden consultarse con:

```powershell
python -m src.analysis.build_contextual_validation_sample --help
python -m src.analysis.evaluate_contextual_validation --help
```

## Diseño de la muestra

Se utilizó una muestra piloto estratificada de 250 artículos, distribuida entre los siguientes grupos:

| Estrato | Número de artículos | Función |
|---|---:|---|
| Contextual de evidencia alta | 60 | Estimar la precisión del grupo candidato a corpus principal |
| Contextual de evidencia media | 80 | Evaluar el grupo reservado para análisis de sensibilidad |
| Caso ambiguo | 40 | Comprobar que los casos con evidencia insuficiente no se incorporen automáticamente |
| Control de caso explícito | 30 | Detectar falsos positivos entre artículos con terminología explícita |
| Control temático explícito | 20 | Diferenciar artículos sobre el tema de artículos sobre casos concretos |
| Negativo difícil | 20 | Detectar posibles omisiones dentro de registros inicialmente descartados |

La muestra se distribuye, en la medida permitida por los datos disponibles, por país, medio, año y categoría de clasificación.

La validación es una validación piloto o de desarrollo. No constituye un conjunto de prueba independiente separado del proceso de ajuste de reglas.

## Campos de anotación manual

Las columnas mínimas son:

### `manual_case_label`

Valores permitidos:

- `relevant`: el artículo contiene un caso pertinente para el alcance del proyecto.
- `not_relevant`: el artículo no contiene un caso pertinente.
- `uncertain`: el contenido disponible no permite tomar una decisión segura.

### `manual_gender_recognition`

Valores permitidos:

- `explicit`: el artículo nombra explícitamente el femicidio, feminicidio,violencia de género, violencia machista u otra denominación equivalente.
- `contextual`: el componente de género se infiere mediante la descripción del caso, la relación entre víctima y agresor, antecedentes, control, violencia sexual u otros indicios.
- `insufficient`: la información no permite identificar con suficiente seguridad el componente de género.

### `manual_notes`

Campo opcional. Solo se utiliza cuando es necesario explicar una decisión ambigua, un error de extracción o una particularidad del artículo.

Otros campos diagnósticos, como el tipo de error o el tipo de violencia, pueden añadirse de forma opcional, pero no son necesarios para calcular las métricas principales.

## Protocolo de anotación

La decisión manual debe basarse en el contenido recuperado del artículo y no únicamente en la categoría automática.

Se recomienda el siguiente orden:

1. Comprobar si el registro corresponde a un artículo periodístico individual.
2. Determinar si describe un caso concreto o solamente aborda el tema de forma general.
3. Identificar si existe una víctima mujer o una población incluida en el alcance definido por el proyecto.
4. Comprobar si existe violencia, amenaza, desaparición, violencia sexual, control, antecedentes u otros elementos relevantes.
5. Determinar si el componente de género es explícito, contextual o insuficiente.
6. Utilizar `uncertain` cuando el texto extraído sea demasiado incompleto o contradictorio.

Las páginas temáticas, índices, portadas o listados de noticias no se consideran artículos individuales, aunque contengan fragmentos de noticias pertinentes.

## Métricas

La evaluación calcula, como mínimo:

- número de artículos por estrato;
- distribución de `relevant`, `not_relevant` y `uncertain`;
- precisión por estrato;
- distribución de reconocimiento explícito, contextual e insuficiente;
- comparación entre la clasificación automática y la decisión manual.

Para el cálculo de precisión por estrato se utiliza:

```text
precision = relevant / (relevant + not_relevant)
```

Los artículos etiquetados como `uncertain` se informan por separado y no se incluyen en el denominador principal.

El grupo de negativos difíciles no permite estimar el recall global del sistema. Su función es detectar omisiones plausibles dentro de un subconjunto deliberadamente complejo.

## Interpretación de la validación piloto

La validación mostró una diferencia clara entre los niveles automáticos:

- los candidatos contextuales de evidencia alta presentan una precisión suficiente para incorporarse al corpus principal;
- los candidatos contextuales de evidencia media presentan mayor incertidumbre y se conservan para análisis de sensibilidad;
- los casos ambiguos contienen una proporción elevada de registros sin información suficiente y no se incorporan a los resultados principales;
- los controles explícitos confirman que la presencia de terminología de género no garantiza por sí sola que el registro corresponda a un caso individual.

Esta interpretación se utiliza para definir los corpus analíticos, no para eliminar los registros originales.

## Política de construcción del corpus final

### Corpus principal

Incluye:

- `candidate_explicit_case`
- `candidate_contextual_high`

Archivo generado:

```text
outputs/final/case_articles_main.parquet
```

### Corpus de sensibilidad

Incluye:

- `review_contextual_medium`

Archivo generado:

```text
outputs/final/case_articles_sensitivity_medium.parquet
```

### Corpus ambiguo

Incluye:

- `review_ambiguous_case`

Archivo generado:

```text
outputs/final/case_articles_ambiguous.parquet
```

El corpus ambiguo se conserva para auditoría, pero no se utiliza en los resultados principales.

## Revisión separada de la dirección de la violencia

La validación contextual y la revisión de posibles agresoras mujeres son dos procesos distintos.

Las reglas automáticas de dirección de la violencia únicamente generan alertas. No eliminan artículos de forma automática.

Las decisiones manuales se almacenan de manera compacta en:

```text
data/annotations/female_aggressor_scope_decisions.csv
```

con las columnas:

```text
article_key
manual_scope_decision
```

Valores permitidos:

- `include`
- `exclude`
- `review`

Solo una decisión manual `exclude` elimina un artículo del corpus
correspondiente.

## Resultado del corpus 2015-2018

Después de aplicar la política de clasificación y las decisiones manuales de
alcance, el periodo 2015-2018 quedó distribuido de la siguiente manera:

| Conjunto | Artículos |
|---|---:|
| Corpus principal | 831 |
| Contextuales medios para sensibilidad | 106 |
| Casos ambiguos | 542 |
| Exclusiones manuales | 18 |
| Inclusiones manuales confirmadas | 18 |
| Alertas pendientes | 0 |

Distribución temporal del corpus principal:

| Año de archivo | Artículos |
|---|---:|
| 2015 | 180 |
| 2016 | 293 |
| 2017 | 219 |
| 2018 | 139 |
| **Total** | **831** |

El año utilizado en estas tablas es `archive_year`, correspondiente al año de la captura recuperada de Wayback Machine. No debe interpretarse automáticamente como la fecha exacta de publicación original.

## Reproducibilidad

Para reconstruir la muestra:

```powershell
python -m src.analysis.build_contextual_validation_sample
```

Para evaluar las anotaciones:

```powershell
python -m src.analysis.evaluate_contextual_validation
```

Para reconstruir el corpus final:

```powershell
python -m src.analysis.build_final_case_corpus
```

Para comprobar el resumen:

```powershell
python -c "import pandas as pd; p=r'outputs\\final\\final_case_corpus_summary.csv'; print(pd.read_csv(p).to_string(index=False))"
```

## Limitaciones

1. La muestra es estratificada y no representa una muestra aleatoria simple de todos los artículos recuperados.
2. La validación es piloto y fue utilizada también para ajustar reglas, por lo que no equivale a una evaluación independiente definitiva.
3. Los textos archivados pueden estar incompletos, contaminados por elementos de navegación o afectados por errores de extracción.
4. `archive_year` representa el año de captura y no necesariamente el año de publicación.
5. La unidad de análisis es el artículo, no el acontecimiento. Varias noticias pueden referirse al mismo caso.
6. Las diferencias absolutas entre años y medios deben interpretarse junto con la cobertura desigual de Wayback Machine.
7. La revisión de 2015-2018 no elimina la necesidad de realizar una pequeña comprobación de estabilidad al aplicar las reglas a años posteriores.

## Aplicación a años posteriores

La misma política puede aplicarse a 2019-2025.

No es necesario repetir una validación completa de 250 artículos para cada
periodo. Se recomienda:

1. aplicar las mismas reglas sin modificarlas;
2. revisar las nuevas alertas de dirección;
3. comprobar una muestra pequeña de casos explícitos, contextuales altos y contextuales medios de los años nuevos;
4. documentar cualquier cambio de vocabulario o comportamiento observado.

Las decisiones manuales deben mantenerse en un archivo acumulativo identificado
por `article_key`.
