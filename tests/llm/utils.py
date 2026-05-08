from src.llm.code_loading import load_function_from_source
from src.llm.prompt_schema import PromptSchema

def run_model_code(code: str, data: dict, params: dict):
    func = load_function_from_source(code, "model")
    return func(data, params)

def run_param_est_code(code: str, data: dict):
    func = load_function_from_source(code, "parameter_estimator")
    return func(data)
