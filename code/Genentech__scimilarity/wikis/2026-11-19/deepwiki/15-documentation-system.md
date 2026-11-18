# Documentation System

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.github/workflows/pages.yaml](.github/workflows/pages.yaml)
- [docs/_static/custom.css](docs/_static/custom.css)
- [docs/_static/training_thumbnail.png](docs/_static/training_thumbnail.png)
- [docs/conf.py](docs/conf.py)

</details>



## Purpose and Scope

The SCimilarity documentation system provides comprehensive technical documentation for the library using Sphinx, including API reference, tutorials, and user guides. This page covers the documentation build infrastructure, Sphinx configuration, automated deployment via GitHub Actions, and the integration of Jupyter notebook tutorials.

For information about the actual content and structure of tutorials, see [Tutorials & Examples](#8). For development workflows and CI/CD beyond documentation, see [Development & Deployment](#7).

## Documentation Architecture

The SCimilarity documentation system uses Sphinx as the primary documentation generator, configured to automatically build API documentation from docstrings, render Jupyter notebook tutorials, and deploy to GitHub Pages. The system integrates multiple content types into a unified documentation site.

```mermaid
graph TB
    subgraph "Content Sources"
        DOCSTRINGS["Python Docstrings"]
        NOTEBOOKS["Jupyter Notebooks"]
        RST["RST Files"]
        STATIC["Static Assets"]
    end
    
    subgraph "Sphinx Build System"
        CONF["docs/conf.py"]
        AUTODOC["sphinx.ext.autodoc"]
        NBSPHINX["nbsphinx"]
        NAPOLEON["sphinx.ext.napoleon"]
        THEME["pydata_sphinx_theme"]
    end
    
    subgraph "Build Pipeline"
        GHACTIONS[".github/workflows/pages.yaml"]
        CONTAINER["ghcr.io/genentech/scimilarity:latest"]
        SPHINXBUILD["sphinx-build"]
        BUILD["_build Directory"]
    end
    
    subgraph "Deployment"
        PAGES["GitHub Pages"]
        ARTIFACT["upload-pages-artifact"]
        DEPLOY["deploy-pages"]
    end
    
    DOCSTRINGS --> AUTODOC
    NOTEBOOKS --> NBSPHINX
    RST --> CONF
    STATIC --> CONF
    
    AUTODOC --> SPHINXBUILD
    NBSPHINX --> SPHINXBUILD
    NAPOLEON --> SPHINXBUILD
    THEME --> SPHINXBUILD
    CONF --> SPHINXBUILD
    
    GHACTIONS --> CONTAINER
    CONTAINER --> SPHINXBUILD
    SPHINXBUILD --> BUILD
    BUILD --> ARTIFACT
    ARTIFACT --> DEPLOY
    DEPLOY --> PAGES
```

Sources: [.github/workflows/pages.yaml:1-52](), [docs/conf.py:1-294]()

## Sphinx Configuration

The Sphinx configuration in `docs/conf.py` defines the comprehensive setup for building SCimilarity documentation, including extensions, theme selection, and content processing options.

### Core Extensions and Features

| Extension | Purpose | Configuration |
|-----------|---------|---------------|
| `sphinx.ext.autodoc` | Auto-generate API docs from docstrings | `autodoc_typehints = "description"` |
| `sphinx.ext.napoleon` | Parse Google/NumPy style docstrings | Enabled by default |
| `nbsphinx` | Render Jupyter notebooks | Custom thumbnails configured |
| `sphinx.ext.intersphinx` | Cross-reference external docs | Currently disabled |
| `sphinx.ext.viewcode` | Add source code links | Enabled |
| `sphinx.ext.mathjax` | Render mathematical expressions | Enabled |

```mermaid
graph LR
    subgraph "autodoc_mock_imports"
        ANNDATA["anndata"]
        HNSWLIB["hnswlib"] 
        TORCH["torch"]
        TILEDB["tiledb"]
        PYTORCH["pytorch_lightning"]
        NUMPY["numpy"]
        PANDAS["pandas"]
        OTHERS["..."]
    end
    
    subgraph "Sphinx Extensions"
        AUTODOC["sphinx.ext.autodoc"]
        NAPOLEON["sphinx.ext.napoleon"]
        NBSPHINX["nbsphinx"]
        VIEWCODE["sphinx.ext.viewcode"]
    end
    
    subgraph "Build Configuration"
        PROJECT["project = 'scimilarity'"]
        THEME["html_theme = 'pydata_sphinx_theme'"]
        STATIC["html_static_path = ['_static']"]
        CSS["html_css_files = ['custom.css']"]
    end
    
    ANNDATA --> AUTODOC
    TORCH --> AUTODOC
    TILEDB --> AUTODOC
    PYTORCH --> AUTODOC
    
    AUTODOC --> PROJECT
    NAPOLEON --> PROJECT
    NBSPHINX --> PROJECT
    VIEWCODE --> PROJECT
    
    PROJECT --> THEME
    THEME --> STATIC
    STATIC --> CSS
```

Sources: [docs/conf.py:30-43](), [docs/conf.py:47-66](), [docs/conf.py:94-95](), [docs/conf.py:160]()

### Notebook Integration

The documentation system uses `nbsphinx` to render Jupyter notebooks as part of the documentation, with custom thumbnail configuration for tutorial notebooks.

```mermaid
graph TB
    subgraph "nbsphinx_thumbnails Configuration"
        SEARCH1["notebooks/cell_search_tutorial_1"]
        SEARCH2["notebooks/cell_search_tutorial_2"] 
        SEARCH3["notebooks/cell_search_tutorial_3"]
        ANNOTATION["notebooks/cell_annotation_tutorial"]
        ADVANCED["notebooks/advanced_tutorial"]
        TRAINING["notebooks/training_tutorial"]
    end
    
    subgraph "Thumbnail Assets"
        THUMB1["_static/search_thumbnail_1.png"]
        THUMB2["_static/search_thumbnail_2.png"]
        THUMB3["_static/search_thumbnail_3.png"]
        THUMBANNOT["_static/annotation_thumbnail.png"]
        THUMBADV["_static/advanced_thumbnail.png"]
        THUMBTRAIN["_static/training_thumbnail.png"]
    end
    
    SEARCH1 --> THUMB1
    SEARCH2 --> THUMB2
    SEARCH3 --> THUMB3
    ANNOTATION --> THUMBANNOT
    ADVANCED --> THUMBADV
    TRAINING --> THUMBTRAIN
```

Sources: [docs/conf.py:72-79]()

## Automated Deployment Pipeline

The documentation deployment uses GitHub Actions to automatically build and publish documentation to GitHub Pages whenever changes are pushed to the main branch.

### Build Workflow Components

```mermaid
graph TB
    subgraph "Trigger Events"
        PUSH["push to main branch"]
        MANUAL["workflow_dispatch"]
    end
    
    subgraph "Build Job"
        CONTAINER["ghcr.io/genentech/scimilarity:latest"]
        CHECKOUT["actions/checkout@v3"]
        DEPS["pip install -r docs/requirements.txt"]
        PAGES["actions/configure-pages@v3"]
        SPHINX["sphinx-build -b html docs _build"]
        UPLOAD["actions/upload-pages-artifact@v3"]
    end
    
    subgraph "Deploy Job"
        DEPLOY["actions/deploy-pages@v4"]
        GHPAGES["GitHub Pages"]
    end
    
    PUSH --> CONTAINER
    MANUAL --> CONTAINER
    
    CONTAINER --> CHECKOUT
    CHECKOUT --> DEPS
    DEPS --> PAGES
    PAGES --> SPHINX
    SPHINX --> UPLOAD
    
    UPLOAD --> DEPLOY
    DEPLOY --> GHPAGES
```

Sources: [.github/workflows/pages.yaml:3-6](), [.github/workflows/pages.yaml:22-40](), [.github/workflows/pages.yaml:42-51]()

### Deployment Configuration

| Configuration | Value | Purpose |
|---------------|-------|---------|
| Container Image | `ghcr.io/genentech/scimilarity:latest` | Provides pre-built environment with dependencies |
| Build Command | `sphinx-build -b html docs _build` | Generates HTML documentation |
| Output Directory | `_build` | Target directory for generated documentation |
| Permissions | `contents: read, pages: write, id-token: write` | Required GitHub token permissions |

Sources: [.github/workflows/pages.yaml:8-12](), [.github/workflows/pages.yaml:25](), [.github/workflows/pages.yaml:35-36](), [.github/workflows/pages.yaml:40]()

## File Structure and Components

The documentation system follows a standard Sphinx project structure with custom configurations for SCimilarity-specific needs.

### Documentation Directory Structure

```mermaid
graph TB
    subgraph "docs/ Directory"
        CONF["conf.py - Sphinx configuration"]
        REQTXT["requirements.txt - Build dependencies"]
        STATIC["_static/ - Static assets"]
        TEMPLATES["_templates/ - Custom templates"]
        RST["*.rst - Documentation source files"]
    end
    
    subgraph "_static/ Assets"
        CSS["custom.css - Custom styling"]
        THUMBNAILS["*.png - Notebook thumbnails"]
        IMAGES["Other static assets"]
    end
    
    subgraph "Generated Output"
        BUILD["_build/ - Generated HTML"]
        DOCTREES[".doctrees/ - Sphinx cache"]
    end
    
    CONF --> BUILD
    STATIC --> CSS
    STATIC --> THUMBNAILS
    CSS --> BUILD
    THUMBNAILS --> BUILD
    RST --> BUILD
```

Sources: [docs/conf.py:82](), [docs/conf.py:192](), [docs/_static/custom.css:1-4]()

### Version Management

The documentation system automatically extracts version information from the package, with fallback to environment variables for Read the Docs compatibility.

```mermaid
graph LR
    subgraph "Version Resolution"
        PACKAGE["scimilarity.__version__"]
        FALLBACK["os.getenv('READTHEDOCS_VERSION')"]
        DEFAULT["'unknown'"]
    end
    
    subgraph "Version Usage"
        VERSION["version variable"]
        RELEASE["release variable"]
        DOCS["Documentation metadata"]
    end
    
    PACKAGE --> VERSION
    FALLBACK --> VERSION
    DEFAULT --> VERSION
    
    VERSION --> RELEASE
    VERSION --> DOCS
    RELEASE --> DOCS
```

Sources: [docs/conf.py:105-113]()

## Build Process and Integration

The documentation build process integrates multiple content sources and processing steps to generate the final documentation site.

### Content Processing Pipeline

```mermaid
graph TB
    subgraph "Source Processing"
        SYSPATH["sys.path.insert(0, '../src')"]
        MOCKIMPORTS["autodoc_mock_imports"]
        EXCLUDE["exclude_patterns"]
    end
    
    subgraph "Extension Processing"
        AUTODOC_EXT["Process Python docstrings"]
        NBSPHINX_EXT["Convert Jupyter notebooks"]
        NAPOLEON_EXT["Parse docstring formats"]
        VIEWCODE_EXT["Generate source links"]
    end
    
    subgraph "Theme and Assets"
        PYDATA_THEME["pydata_sphinx_theme"]
        CUSTOM_CSS["custom.css"]
        STATIC_FILES["Static asset copying"]
    end
    
    subgraph "Output Generation"
        HTML_BUILD["HTML file generation"]
        SEARCH_INDEX["Search index creation"]
        CROSS_REFS["Cross-reference resolution"]
    end
    
    SYSPATH --> AUTODOC_EXT
    MOCKIMPORTS --> AUTODOC_EXT
    EXCLUDE --> NBSPHINX_EXT
    
    AUTODOC_EXT --> HTML_BUILD
    NBSPHINX_EXT --> HTML_BUILD
    NAPOLEON_EXT --> HTML_BUILD
    VIEWCODE_EXT --> HTML_BUILD
    
    PYDATA_THEME --> HTML_BUILD
    CUSTOM_CSS --> HTML_BUILD
    STATIC_FILES --> HTML_BUILD
    
    HTML_BUILD --> SEARCH_INDEX
    HTML_BUILD --> CROSS_REFS
```

Sources: [docs/conf.py:20](), [docs/conf.py:47-66](), [docs/conf.py:127](), [docs/conf.py:160](), [docs/conf.py:195]()

---