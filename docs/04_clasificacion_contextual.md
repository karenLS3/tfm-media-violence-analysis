# Capa de clasificación contextual

## Objetivo

Separar dos dimensiones que no deben confundirse:

1. **Recuperación** (`retrieval_bucket`): determina qué noticias entran como candidatas mediante reglas amplias.
2. **Reconocimiento del componente de género**: determina si el medio nombra explícitamente el hecho o si la posible dimensión de género solo aparece mediante el contexto.

La capa contextual no modifica `retrieval_bucket`. Produce columnas analíticas adicionales y conserva la clasificación de recuperación como línea base.

## Variables principales

### `explicit_label_type`

- `femicide_feminicide`
- `gender_violence`
- `gendered_description`
- `none`

### `contextual_evidence_level`

- `high`: víctima femenina, hecho violento y relación de pareja o expareja, antecedentes de violencia, control, violencia sexual u otra combinación contextual fuerte dentro de una ventana próxima.
- `medium`: víctima femenina y hecho violento acompañados por un posible agresor masculino, una relación familiar o cercana, una denuncia o un delito genérico con vínculo íntimo, pero sin evidencia suficiente para asignar un nivel alto.
- `low`: muerte, lesión, desaparición o referencia a violencia contra una mujersin contexto suficiente para establecer un caso con seguridad.
- `none`: no se identificó una combinación contextual útil.

La presencia aislada de una palabra como `mujer`, `abuso`, `hijo`, `pareja` o `muerte` no determina por sí sola la inclusión de un artículo. La clasificación depende de la relación contextual entre los indicios.

### `gender_recognition_mode`

- `explicit_femicide_feminicide`
- `explicit_gender_violence`
- `contextual_without_explicit_label`
- `gendered_description_without_explicit_label`
- `ambiguous_case_without_explicit_label`
- `no_case_evidence`

### `provisional_case_status`

- `candidate_explicit_case`
- `candidate_contextual_high`
- `review_contextual_medium`
- `review_ambiguous_case`
- `topic_explicit_not_case`
- `not_selected`

Los estados son generados automáticamente y se consideran provisionales hasta aplicar la política de construcción del corpus final.

La clasificación fue evaluada mediante una validación piloto estratificada. A partir de esa validación se adoptó la siguiente política:

- `candidate_explicit_case` y `candidate_contextual_high` se incorporan al corpus principal;
- `review_contextual_medium` se conserva para análisis de sensibilidad;
- `review_ambiguous_case` se mantiene separado y no se utiliza en los resultados principales;
- `topic_explicit_not_case` y `not_selected` no se incorporan al corpus de casos.

## Niveles de datos

Se conservan dos datasets:

1. `articles_contextual_snapshots.parquet`: todas las capturas de Wayback Machine. Sirve para estudiar actualizaciones de una misma URL.
2. `articles_contextual_unique.parquet`: una fila por artículo, medio y año. Sirve para totales comparables sin contar repetidamente la misma URL.

La agrupación de artículos diferentes que cubren un mismo acontecimiento requiere una fase posterior de identificación de eventos y no debe confundirse con la deduplicación por URL.
La unidad de análisis del corpus actual es el artículo periodístico, no el acontecimiento.

## Revisión manual mínima

No es necesario revisar todo el corpus. La validación se concentra en una muestra estratificada por nivel de evidencia y grupo de control.

La muestra piloto utilizada contiene 250 artículos para el rango de años 2015-2018:

| Estrato | Artículos | Función |
|---|---:|---|
| Contextual de evidencia alta | 60 | Evaluar el grupo candidato al corpus principal |
| Contextual de evidencia media | 80 | Evaluar el grupo de sensibilidad |
| Caso ambiguo | 40 | Comprobar el comportamiento de los casos con evidencia insuficiente |
| Control de caso explícito | 30 | Detectar falsos positivos entre los casos explícitos |
| Control temático explícito | 20 | Diferenciar artículos temáticos de casos individuales |
| Negativo difícil | 20 | Detectar omisiones plausibles entre registros descartados |
| **Total** | **250** | |

La muestra se distribuye, en la medida permitida por los datos disponibles, por país, medio, año y categoría automática.

La validación se considera piloto o de desarrollo. No constituye una prueba independiente completamente separada del ajuste de reglas.

## Campos de validación manual

Los campos mínimos son:

### `manual_case_label`

Valores permitidos:

- `relevant`
- `not_relevant`
- `uncertain`

### `manual_gender_recognition`

Valores permitidos:

- `explicit`
- `contextual`
- `insufficient`

### `manual_notes`

Campo opcional. Se utiliza únicamente cuando es necesario documentar:

- un texto incompleto;
- una decisión incierta;
- una página temática o índice;
- un problema de extracción;
- una particularidad relevante del artículo.

No es obligatorio completar notas para todas las filas.

## Corpus resultantes

### Corpus principal

Incluye:

- `candidate_explicit_case`
- `candidate_contextual_high`

Archivo:

```text
outputs/final/case_articles_main.parquet
```

Este corpus se utiliza para los resultados principales.

### Corpus de sensibilidad

Incluye:

- `review_contextual_medium`

Archivo:

```text
outputs/final/case_articles_sensitivity_medium.parquet
```

Este corpus se utiliza para comprobar si las conclusiones cambian al incorporar casos con evidencia contextual media.

### Corpus ambiguo

Incluye:

- `review_ambiguous_case`

Archivo:

```text
outputs/final/case_articles_ambiguous.parquet
```

Se conserva para auditoría y posibles revisiones posteriores, pero no forma parte del análisis principal.

## Revisión de la dirección de la violencia

La revisión de posibles agresoras mujeres es una fase distinta de la validación contextual.

Las reglas automáticas de dirección:

- solo generan alertas;
- no eliminan artículos de forma automática;
- pueden producir falsos positivos cuando relacionan menciones pertenecientes a oraciones o fragmentos diferentes.

Las decisiones manuales se almacenan en:

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

Solo:

```text
manual_scope_decision = exclude
```

elimina un artículo del corpus correspondiente.

Una decisión `include` confirma que el artículo debe conservarse. Una decisión `review` mantiene el registro separado o pendiente de resolución.

## Resultado del corpus 2015-2018

Después de aplicar la clasificación contextual y las decisiones manuales de alcance, el periodo 2015-2018 quedó distribuido así:

| Conjunto | Artículos |
|---|---:|
| Corpus principal | 831 |
| Contextuales medios para sensibilidad | 106 |
| Casos ambiguos | 542 |
| Exclusiones manuales | 18 |
| Inclusiones manuales confirmadas | 18 |
| Alertas pendientes | 0 |

Distribución anual del corpus principal:

| Año de archivo | Artículos |
|---|---:|
| 2015 | 180 |
| 2016 | 293 |
| 2017 | 219 |
| 2018 | 139 |
| **Total** | **831** |

El campo `archive_year` corresponde al año de la captura recuperada de Wayback Machine. No debe interpretarse automáticamente como la fecha exacta de publicación original.

## Aplicación a años posteriores

La misma lógica puede aplicarse a 2019-2025 sin modificar las reglas principales.

Para los años nuevos se debe:

1. reconstruir `articles_contextual_unique.parquet`;
2. aplicar `build_final_case_corpus.py`;
3. revisar las nuevas alertas de dirección;
4. mantener acumuladas las decisiones manuales mediante `article_key`;
5. realizar una comprobación pequeña de estabilidad sobre casos explícitos, contextuales altos y contextuales medios.

No es necesario repetir una validación completa de 250 artículos para cada periodo, salvo que se modifiquen sustancialmente las reglas o el vocabulario.

## Documentación relacionada

La metodología y los resultados de la validación se describen con mayor detalle en:

```text
docs/05_validacion_contextual.md
```
