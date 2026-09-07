# Minería de noticias para el análisis del tratamiento mediático de la violencia machista en Argentina y México

Repositorio del Trabajo Final de Máster orientado a la recuperación, procesamiento y análisis computacional de noticias relacionadas con violencia machista, femicidio y feminicidio en medios digitales de Argentina y México.

El proyecto implementa un pipeline reproducible en Python para recuperar contenidos históricos mediante Wayback Machine, extraer texto y metadatos de los artículos, seleccionar documentos potencialmente relevantes y construir posteriormente el corpus utilizado para el análisis.

Repositorio: `tfm-media-violence-analysis`

## Estructura general del pipeline

El procesamiento documental se organiza en las siguientes etapas:

```text
Wayback Machine
      │
      ▼
build_cdx_index
      │
      ▼
build_homepage_candidates
      │
      ▼
build_clean_candidates
      │
      ▼
build_article_texts
      │
      ▼
retry_failed_article_texts
      │
      ▼
build_relevance_dataset
      │
      ▼
clasificación y consolidación del corpus
      │
      ▼
corpus analítico
```

Las ejecuciones se almacenan de forma independiente mediante un identificador `run_id`.

Por ejemplo:

```text
data/runs/2015_01/
data/runs/2015_02/
...
```

Los datos descargados, resultados intermedios y outputs generados no forman parte del código fuente versionado en Git.

## Requisitos

El proyecto utiliza Python y las dependencias definidas en el entorno del repositorio.

Se recomienda trabajar dentro de un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar posteriormente las dependencias correspondientes al proyecto.

## Ejecución de una corrida individual

### 1. Recuperar el índice de capturas de Wayback Machine

```powershell
python -m src.acquisition.build_cdx_index `
    --config configs/sources_argentina_mexico.yml `
    --from-date 20150101 `
    --to-date 20150131 `
    --run-id 2015_01
```

### 2. Extraer enlaces candidatos desde las páginas archivadas

```powershell
python -m src.acquisition.build_homepage_candidates `
    --run-id 2015_01
```

### 3. Limpiar y normalizar los candidatos

```powershell
python -m src.filtering.build_clean_candidates `
    --run-id 2015_01
```

### 4. Descargar y extraer los artículos

```powershell
python -m src.acquisition.build_article_texts `
    --run-id 2015_01 `
    --sleep-seconds 3
```

Esta etapa extrae directamente los campos canónicos del artículo:

```text
title
title_source
publication_date
publication_date_source
text
text_length
```

La extracción de texto y metadatos se realiza en una única pasada mediante `src/extraction/parse_article.py`.

### 5. Reintentar descargas fallidas

```powershell
python -m src.acquisition.retry_failed_article_texts `
    --run-id 2015_01 `
    --sleep-seconds 3
```

### 6. Clasificar los documentos candidatos

```powershell
python -m src.filtering.build_relevance_dataset `
    --run-id 2015_01
```

El resultado principal de esta etapa se guarda dentro de:

```text
data/runs/<run_id>/processed/
```

## Ejecución completa del pipeline

Para ejecutar automáticamente el pipeline por meses:

```powershell
$log = "outputs\logs\pipeline_full_$((Get-Date).ToString('yyyyMMdd_HHmmss')).txt"

Start-Transcript -Path $log

python -m src.pipeline.run_pipeline `
    --mode monthly `
    --start-year 2015 `
    --end-year 2025 `
    --config configs/sources_argentina_mexico_full.yml `
    --continue-on-error `
    --sleep-seconds 3

Stop-Transcript
```

Este procedimiento ejecuta de forma secuencial las etapas de adquisición, extracción, reintentos y clasificación para cada período definido.


## Corpus analítico

Una vez consolidados y seleccionados los documentos que forman parte del estudio, el corpus principal se utiliza como entrada para la fase de análisis.

La construcción del corpus analítico canónico se realiza mediante:

```powershell
python -m src.analysis.build_analysis_corpus `
    --config configs/analysis.yml
```

Entre otras operaciones, esta etapa:

* establece una identidad canónica para cada artículo;
* resuelve duplicados;
* excluye páginas conocidas que no corresponden a artículos;
* determina el año analítico utilizando la mejor información temporal disponible;
* conserva únicamente el texto del artículo para el análisis lingüístico;
* genera auditorías de calidad y trazabilidad.


## Organización del repositorio

```text
configs/
    configuraciones de fuentes y análisis

src/
    acquisition/
        recuperación de contenidos archivados

    extraction/
        extracción de texto y metadatos

    filtering/
        limpieza y clasificación de candidatos

    analysis/
        construcción y análisis del corpus

    pipeline/
        orquestación de ejecuciones

    utils/
        utilidades compartidas

    validation/
        procedimientos de validación

tests/
    pruebas automatizadas

data/
    datos generados durante la ejecución
    no versionados

outputs/
    resultados, auditorías y reportes
    no versionados
```

## Reproducibilidad

El repositorio mantiene únicamente los componentes necesarios para reproducir el procedimiento:

* código fuente;
* configuraciones;
* pruebas automatizadas;
* documentación metodológica;
* archivos de anotación manual necesarios para reproducir decisiones del corpus.

Los siguientes elementos no se almacenan en Git:

* HTML descargado;
* textos completos de artículos;
* archivos Parquet generados;
* ejecuciones contenidas en `data/runs`;
* tablas y gráficos generados en `outputs`;
* archivos temporales o históricos de desarrollo.

## Autoría

Proyecto desarrollado por [@karenL26](https://github.com/karenL26) como parte de un Trabajo Final de Máster.
