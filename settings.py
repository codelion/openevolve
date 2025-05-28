import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
FLASH_API_KEY = os.getenv("FLASH_API_KEY")
FLASH_BASE_URL = os.getenv("FLASH_BASE_URL", None)
FLASH_MODEL = os.getenv("FLASH_MODEL")

# PRO_API_KEY = os.getenv("PRO_API_KEY")
# PRO_BASE_URL = os.getenv("PRO_BASE_URL", None)
# PRO_MODEL = os.getenv("PRO_MODEL")
CUSTOM_PROVIDERS = {
    PRO_KEY: {
        "base_url": os.getenv("PRO_BASE_URL", "http://127.0.0.1:11434"),
        "api_key": os.getenv("PRO_API_KEY", ""),
        "model": os.getenv("PRO_MODEL", "llama3.2:1b"),
        "provider": os.getenv("PRO_PROVIDER", "ollama")
    },
    FLASH_KEY: {
        "base_url": os.getenv("FLASH_BASE_URL", "http://127.0.0.1:11434"),
        "api_key": os.getenv("FLASH_API_KEY", ""),
        "model": os.getenv("FLASH_MODEL", "llama3.2:1b"),
        "provider": os.getenv("FLASH_PROVIDER", "ollama")
    },
    EVALUATION_KEY: {
        "base_url": os.getenv("EVALUATION_BASE_URL", "http://127.0.0.1:11434"),
        "api_key": os.getenv("EVALUATION_API_KEY", ""),
        "model": os.getenv("EVALUATION_MODEL", "llama3.2:1b"),
        "provider": os.getenv("EVALUATION_PROVIDER", "ollama")
    }
}

EVALUATION_API_KEY = os.getenv("EVALUATION_API_KEY")
EVALUATION_BASE_URL = os.getenv("EVALUATION_BASE_URL", None)
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL")

# LiteLLM Configuration
LITELLM_DEFAULT_MODEL = os.getenv("LITELLM_DEFAULT_MODEL", "gpt-3.5-turbo")
LITELLM_DEFAULT_BASE_URL = os.getenv("LITELLM_DEFAULT_BASE_URL", None)
LITELLM_MAX_TOKENS = os.getenv("LITELLM_MAX_TOKENS")
LITELLM_TEMPERATURE = os.getenv("LITELLM_TEMPERATURE")
LITELLM_TOP_P = os.getenv("LITELLM_TOP_P")
LITELLM_TOP_K = os.getenv("LITELLM_TOP_K")

# if not PRO_API_KEY:
#     print("Warning: PRO_API_KEY not found in .env or environment. Using a NON-FUNCTIONAL placeholder. Please create a .env file with your valid API key.")
#     PRO_API_KEY = "Your API key"

# Evolutionary Algorithm Settings
POPULATION_SIZE = 5
GENERATIONS = 2
# Threshold for switching to bug-fix prompt
# If a program has errors and its correctness score is below this, a bug-fix prompt will be used.
BUG_FIX_CORRECTNESS_THRESHOLD = float(os.getenv("BUG_FIX_CORRECTNESS_THRESHOLD", "0.1"))
ELITISM_COUNT = 1
MUTATION_RATE = 0.7
CROSSOVER_RATE = 0.2

# Island Model Settings
NUM_ISLANDS = 4  # Number of subpopulations
MIGRATION_INTERVAL = 4  # Number of generations between migrations
ISLAND_POPULATION_SIZE = POPULATION_SIZE // NUM_ISLANDS  # Programs per island
MIN_ISLAND_SIZE = 2  # Minimum number of programs per island
MIGRATION_RATE = 0.2  # Rate at which programs migrate between islands

# Debug Settings
DEBUG = os.getenv("DEBUG", False)
EVALUATION_TIMEOUT_SECONDS = 800

# Docker Execution Settings
DOCKER_IMAGE_NAME = os.getenv("DOCKER_IMAGE_NAME", "code-evaluator:latest")
DOCKER_NETWORK_DISABLED = os.getenv("DOCKER_NETWORK_DISABLED", "True").lower() == "true"

DATABASE_TYPE = "in_memory"
DATABASE_PATH = "program_database.json"

# Logging Configuration
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"
LOG_FILE = "alpha_evolve.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10
API_TIMEOUT = 5
API_TIMEOUT_CONNECT = float(os.getenv("API_TIMEOUT_CONNECT", "5.0"))
API_TIMEOUT_READ    = float(os.getenv("API_TIMEOUT_READ",    "60.0"))
API_TIMEOUT_WRITE   = float(os.getenv("API_TIMEOUT_WRITE",   "60.0"))
API_TIMEOUT_POOL    = float(os.getenv("API_TIMEOUT_POOL",    "60.0"))
API_RETRY_DELAY_SECONDS = 10 # Initial delay, will be exponential

RL_TRAINING_INTERVAL_GENERATIONS = 50
RL_MODEL_PATH = "rl_finetuner_model.pth"

MONITORING_DASHBOARD_URL = "http://localhost:8080"

def get_setting(key, default=None):
    """
    Retrieves a setting value.
    For LLM models, it specifically checks if the primary choice is available,
    otherwise falls back to a secondary/default if defined.
    """
    return globals().get(key, default)

def get_llm_model(model_type="default"):
    if model_type == "default":
        return LITELLM_DEFAULT_MODEL
    elif model_type == "flash":
        # Assuming FLASH_MODEL might still be a specific, different model.
        # If FLASH_MODEL is also meant to be covered by litellm's general handling,
        # this could also return LITELLM_DEFAULT_MODEL or a specific flash model string.
        # For now, keep FLASH_MODEL if it's distinct.
        return FLASH_MODEL if FLASH_MODEL else LITELLM_DEFAULT_MODEL # Return default if FLASH_MODEL is not set
    # Fallback for any other model_type not explicitly handled
    return LITELLM_DEFAULT_MODEL

                                 
