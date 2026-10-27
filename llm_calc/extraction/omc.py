"""
Side-effect-free access to the OpenMedCalc request models and calculators.

Importing ``llm_calc.tools.openmedcalc`` starts a FastAPI app, builds LLM clients
and opens the experiment database, which needs API keys and a configured
environment. The extraction benchmark only needs the Pydantic request models and
the pure ``calculate_*`` functions, so this module loads them directly from the
source files:

* ``api_datamodel.py`` is imported under a private module name (it only depends
  on pydantic).
* The ``calculate_*`` functions are lifted out of ``openmedcalc/__init__.py``
  with ``ast``, stripped of their FastAPI route decorators, and compiled into a
  private namespace. The calculator source therefore stays the single source of
  truth: a change to a calculator in OpenMedCalc is picked up here automatically.
"""

import ast
import copy
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Type

import numpy as np
from pydantic import BaseModel

OMC_DIR = Path(__file__).resolve().parents[1] / "tools" / "openmedcalc"

# calculator slug -> (request model name, calculate function name)
CALCULATORS: Dict[str, tuple] = {
    "caprini": ("CalcRequestCapriniVte", "calculate_caprini"),
    "wellsdvt": ("CalcRequestWellsDvt", "calculate_wells_dvt"),
    "meldna": ("CalcRequestMeldNa", "calculate_meld_na"),
    "sofa": ("CalcRequestSOFA", "calculate_sofa"),
    "cci": ("CalcRequestCCI", "calculate_cci"),
    "ariscat": ("CalcRequestAriscat", "calculate_ariscat"),
    "psiport": ("CalcRequestPsiPort", "calculate_psi_port"),
    "gad7": ("CalcRequestGAD7", "calculate_gad7"),
    "hasbled": ("CalcRequestHASBLED", "calculate_hasbled"),
    "nihss": ("CalcRequestNIHSS", "calculate_nihss"),
}


@lru_cache(maxsize=1)
def datamodel():
    """The ``api_datamodel`` module, imported without its parent package."""
    name = "_omc_api_datamodel"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, OMC_DIR / "api_datamodel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _calculate_functions() -> Dict[str, Callable]:
    source = (OMC_DIR / "__init__.py").read_text()
    tree = ast.parse(source)
    wanted = {fn for _, fn in CALCULATORS.values()}
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            functions.append(node)
    missing = wanted - {f.name for f in functions}
    if missing:
        raise RuntimeError(f"calculate functions not found in OpenMedCalc: {missing}")

    namespace = {name: getattr(datamodel(), name) for name in dir(datamodel())}
    # calc_docs reads markdown that only the web app ships; the score does not use it
    namespace.update({"np": np, "calc_docs": lambda *_: None})
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(OMC_DIR / "__init__.py"), "exec"), namespace)
    return {f.name: namespace[f.name] for f in functions}


def request_model(calculator: str) -> Type[BaseModel]:
    return getattr(datamodel(), CALCULATORS[calculator][0])


def calculate(calculator: str, request: BaseModel) -> int:
    """Score a validated request with the OpenMedCalc implementation."""
    fn = _calculate_functions()[CALCULATORS[calculator][1]]
    # some calculators clamp inputs in place (e.g. MELD-Na); never mutate the caller's copy
    return fn(copy.deepcopy(request)).score
