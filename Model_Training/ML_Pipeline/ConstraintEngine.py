import json
import google.generativeai as genai

# MODULE 2: CONSTRAINT ENGINE

def run_constraint_engine(state, api_key):
    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel("gemini-2.5-flash") 

    input_state_str = json.dumps(state, indent=2)

    prompt = f"""
You are a STRICT AutoML Architect.

You MUST follow ALL rules EXACTLY. Any violation makes the output invalid.

----------------------------------------
INPUT STATE:
{input_state_str}

----------------------------------------
HARD CONSTRAINTS (NON-NEGOTIABLE):

DATASET SIZE RULES:

1. VERY SMALL DATASET (num_samples < 1000):
   - num_hidden_layers: low=1, high=2
   - num_neurons_base: low=16, high=128

2. SMALL/MEDIUM DATASET (num_samples >= 1000):
   - num_hidden_layers: low=2, high=6
   - num_neurons_base: low=32, high=256

BATCH SIZE RULES:
   - num_samples < 1000  → choices: [8, 16, 32]
   - num_samples >= 1000 AND compute=gpu  → choices: [64, 128, 256]
   - num_samples >= 1000 AND compute=cpu  → choices: [16, 32, 64]

OTHER CONSTRAINTS:
   - activation_function choices: ["relu", "gelu","tanh","sigmoid","softmax","linear"]
   - optimizer choices: ["adam", "adamw","SGD"]
   - scheduler choices: ["none", "cosine"]
   - dropout_rate: medium/high noise → low=0.2, high=0.5 | low noise → low=0.1, high=0.3
   - epochs: low=30, high=80
   - early_stopping_patience: low=5, high=12
   - grad_clip: low=0.5, high=3.0

IMBALANCE RULE:
   - If is_balanced=false: dropout_rate low >= 0.2, weight_decay low >= 1e-5

OUTPUT FORMAT:
STRICT JSON ONLY. No markdown, no explanation.
Must be a completely FLAT dictionary inside "search_space". NO NESTING of layers/network.
Only use valid keys: "type", "low", "high", "choices", "log". NEVER use "_type" or "_low".
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    # SAFE PARSE
    def safe_parse(text: str):
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                print("⚠️ No JSON detected in LLM response — using default space.")
                return None
            parsed = json.loads(text[start:end])
            if "search_space" not in parsed:
                print("⚠️ LLM JSON missing 'search_space' key — using default space.")
                return None
            return parsed
        except Exception as e:
            print(f"⚠️ JSON parse failed ({e}) — using default space.")
            return None

    # DEFAULT BASE SPACE (Cleaned)
    def default_space() -> dict:
        return {
            "search_space": {
                "num_hidden_layers":       {"type": "int",         "low": 2,    "high": 6},
                "num_neurons_base":        {"type": "int",         "low": 32,   "high": 256,  "log": True},
                "layer_shrink_factor":     {"type": "float",       "low": 0.5,  "high": 1.0},
                "activation_function":     {"type": "categorical", "choices": ["relu", "gelu","tanh","sigmoid","softmax","linear"]},
                "use_batch_norm":          {"type": "categorical", "choices": [True, False]},
                "dropout_rate":            {"type": "float",       "low": 0.1,  "high": 0.5},
                "weight_decay":            {"type": "float",       "low": 1e-6, "high": 1e-2, "log": True},
                "optimizer":               {"type": "categorical", "choices": ["adam", "adamw","SGD"]},
                "beta1":                   {"type": "float",       "low": 0.85, "high": 0.99},
                "learning_rate":           {"type": "float",       "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size":              {"type": "categorical", "choices": [64, 128, 256]},
                "epochs":                  {"type": "int",         "low": 30,   "high": 80},
                "early_stopping_patience": {"type": "int",         "low": 5,    "high": 12},
                "grad_clip":               {"type": "float",       "low": 0.5,  "high": 3.0},
                "scheduler":               {"type": "categorical", "choices": ["none", "cosine"]},
                "weight_init":             {"type": "categorical", "choices": ["xavier", "kaiming"]},
            }
        }

    # ENSURE SCHEMA 
    def ensure_schema(ss: dict) -> dict:
        defaults = default_space()["search_space"]
        cleaned_ss = {}

        for key, default_val in defaults.items():
            cleaned_ss[key] = default_val.copy()
            user_val = ss.get(key, {})
            if isinstance(user_val, dict):
                t = user_val.get("type", user_val.get("_type"))
                if t: cleaned_ss[key]["type"] = t

                low = user_val.get("low", user_val.get("_low"))
                if low is not None: cleaned_ss[key]["low"] = low

                high = user_val.get("high", user_val.get("_high"))
                if high is not None: cleaned_ss[key]["high"] = high

                choices = user_val.get("choices", user_val.get("_choices"))
                if choices is not None: cleaned_ss[key]["choices"] = choices

                log = user_val.get("log", user_val.get("_log"))
                if log is not None: cleaned_ss[key]["log"] = log

        return cleaned_ss

    # ENFORCE CONSTRAINTS
    def enforce_constraints(space: dict | None, state_meta: dict) -> dict:
        if space is None or "search_space" not in space:
            space = default_space()

        ss = ensure_schema(space["search_space"])

        meta    = state_meta["dataset_meta"]
        compute = state_meta["constraints"]["compute"]
        prefs   = state_meta["preferences"]

        num_samples   = meta["num_samples"]
        noise         = meta["noise_level"]
        is_imbalanced = not meta["is_balanced"]
        problem_type  = state_meta["problem_type"]

        if num_samples < 1000:
            ss["num_hidden_layers"]["low"]  = 1
            ss["num_hidden_layers"]["high"] = 2
            ss["num_neurons_base"]["low"]   = 16
            ss["num_neurons_base"]["high"]  = 64
            ss["dropout_rate"]["low"]       = max(0.2, ss["dropout_rate"]["low"])
            ss["dropout_rate"]["high"]      = min(0.4, ss["dropout_rate"]["high"])
        else:
            ss["num_hidden_layers"]["low"]  = 2
            ss["num_hidden_layers"]["high"] = 6
            ss["num_neurons_base"]["low"]   = 32
            ss["num_neurons_base"]["high"]  = 256

        if num_samples < 1000:
            ss["batch_size"]["choices"] = [8, 16, 32]
        elif compute == "gpu":
            ss["batch_size"]["choices"] = [64, 128, 256]
        else:
            ss["batch_size"]["choices"] = [16, 32, 64]

        if noise == "low":
            ss["dropout_rate"]["low"]  = max(0.1, ss["dropout_rate"]["low"])
            ss["dropout_rate"]["high"] = min(0.3, ss["dropout_rate"]["high"])
        else:
            ss["dropout_rate"]["low"]  = max(0.2, ss["dropout_rate"]["low"])
            ss["dropout_rate"]["high"] = min(0.5, ss["dropout_rate"]["high"])

        if is_imbalanced:
            ss["dropout_rate"]["low"]  = max(0.2, ss["dropout_rate"]["low"])
            ss["weight_decay"]["low"]  = max(1e-5, ss["weight_decay"]["low"])

        complexity = prefs.get("model_complexity", "medium")
        if complexity == "high":
            ss["num_hidden_layers"]["high"] = min(ss["num_hidden_layers"]["high"] + 1, 6)
            ss["num_neurons_base"]["high"]  = min(ss["num_neurons_base"]["high"] * 2, 512)
        elif complexity == "low":
            ss["num_hidden_layers"]["high"] = min(ss["num_hidden_layers"]["high"], 2)
            ss["num_neurons_base"]["high"]  = min(ss["num_neurons_base"]["high"], 128)

        ss["activation_function"]["choices"] = ["relu", "gelu","tanh","softmax","sigmoid","linear"]
        ss["optimizer"]["choices"]           = ["adam", "adamw","SGD"]
        ss["scheduler"]["choices"]           = ["none", "cosine"]

        if problem_type == "regression":
            ss["weight_init"]["choices"] = ["kaiming", "xavier"]

        ss["epochs"]["low"]  = max(30, ss["epochs"].get("low", 30))
        ss["epochs"]["high"] = min(80, ss["epochs"].get("high", 80))

        ss["early_stopping_patience"]["low"]  = max(5,  ss["early_stopping_patience"].get("low", 5))
        ss["early_stopping_patience"]["high"] = min(12, ss["early_stopping_patience"].get("high", 12))

        ss["grad_clip"]["low"]  = max(0.5, ss["grad_clip"].get("low", 0.5))
        ss["grad_clip"]["high"] = min(3.0, ss["grad_clip"].get("high", 3.0))

        if ss["num_hidden_layers"]["low"] >= ss["num_hidden_layers"]["high"]:
            ss["num_hidden_layers"]["low"] = max(1, ss["num_hidden_layers"]["high"] - 1)

        if ss["num_neurons_base"]["low"] >= ss["num_neurons_base"]["high"]:
            ss["num_neurons_base"]["low"] = max(16, ss["num_neurons_base"]["high"] // 2)

        if ss["dropout_rate"]["low"] >= ss["dropout_rate"]["high"]:
            ss["dropout_rate"]["low"] = max(0.0, ss["dropout_rate"]["high"] - 0.1)

        return {"search_space": ss}

    parsed      = safe_parse(text)
    final_space = enforce_constraints(parsed, state)

    return final_space