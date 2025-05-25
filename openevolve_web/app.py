from flask import Flask, render_template, jsonify, request
import threading
import os
import yaml
import subprocess
import uuid
import logging # For better logging

# --- Constants ---
# Evolution Status Strings
STATUS_PREPARING = 'preparing'
STATUS_STARTING = 'starting'
STATUS_RUNNING = 'running'
STATUS_COMPLETED = 'completed'
STATUS_FAILED = 'failed'
STATUS_NOT_FOUND = 'not_found'

# File Paths (relative to project root)
# PROJECT_ROOT is defined after app initialization to use __file__
# These paths will be joined with PROJECT_ROOT later.
WEB_CONFIG_FILENAME = 'openevolve_web/web_config.yaml'
DEFAULT_CONFIG_FILENAME = 'configs/default_config.yaml'
TEMP_EVOLUTION_DIR_NAME = 'temp_evolution_files' # Directory to store temporary files for runs

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Global State ---
evolutions = {} # Stores the state of ongoing and completed evolution runs
# Determine project root. app.py is in openevolve_web, so '..' goes to project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# --- Flask App Enhancements ---
@app.after_request
def add_security_headers(response):
    """Adds basic security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    # response.headers['Content-Security-Policy'] = "default-src 'self'" # Example, more complex
    return response

# --- Helper Functions ---

def get_full_path(relative_path):
    """Constructs an absolute path from a path relative to the project root."""
    return os.path.join(PROJECT_ROOT, relative_path)

def load_web_config():
    """
    Loads web configuration from WEB_CONFIG_FILENAME.
    Falls back to DEFAULT_CONFIG_FILENAME if WEB_CONFIG_FILENAME is not found, empty, or corrupted.
    Falls back to a basic Python dictionary if both are problematic.
    """
    web_config_full_path = get_full_path(WEB_CONFIG_FILENAME)
    default_config_full_path = get_full_path(DEFAULT_CONFIG_FILENAME)
    config_loaded = False

    if os.path.exists(web_config_full_path) and os.path.getsize(web_config_full_path) > 0:
        try:
            with open(web_config_full_path, 'r') as f:
                logging.info(f"Loading configuration from {WEB_CONFIG_FILENAME}")
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.warning(f"Corrupted YAML in {WEB_CONFIG_FILENAME}: {e}. Attempting to load default config.")
        except IOError as e:
            logging.warning(f"IOError reading {WEB_CONFIG_FILENAME}: {e}. Attempting to load default config.")
    
    if os.path.exists(default_config_full_path) and os.path.getsize(default_config_full_path) > 0:
        try:
            with open(default_config_full_path, 'r') as f:
                logging.info(f"Loading configuration from {DEFAULT_CONFIG_FILENAME}")
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.warning(f"Corrupted YAML in {DEFAULT_CONFIG_FILENAME}: {e}. Falling back to basic default config.")
        except IOError as e:
            logging.warning(f"IOError reading {DEFAULT_CONFIG_FILENAME}: {e}. Falling back to basic default config.")

    logging.warning("Using basic default configuration as all YAML files are missing, empty, or corrupted.")
    return {'setting1': 'default_value_basic', 'setting2': 123, 'llm': {'provider': 'default_provider'}} # Ensure some structure

def save_web_config(config_data):
    """Saves the given configuration data to WEB_CONFIG_FILENAME."""
    web_config_full_path = get_full_path(WEB_CONFIG_FILENAME)
    try:
        os.makedirs(os.path.dirname(web_config_full_path), exist_ok=True)
        with open(web_config_full_path, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False, indent=4) # Added indent for readability
        logging.info(f"Configuration saved to {WEB_CONFIG_FILENAME}")
    except IOError as e: # More specific exception
        logging.error(f"Error saving web config to {web_config_full_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving web config: {e}")


# --- API Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML page of the web interface."""
    return render_template('index.html')

@app.route('/api/get_settings', methods=['GET'])
def get_settings_endpoint(): # Renamed to avoid conflict with any potential 'get_settings' var
    """API endpoint to retrieve the current application settings."""
    config = load_web_config()
    return jsonify(config)

@app.route('/api/update_settings', methods=['POST'])
def update_settings_endpoint(): # Renamed
    """API endpoint to update application settings."""
    try:
        data = request.get_json()
        if not data: # Handles null or empty request body
            logging.warning("Update settings attempt with invalid/empty JSON payload.")
            return jsonify({'error': 'Invalid JSON payload. Ensure Content-Type is application/json and body is not empty.'}), 400
        
        if not isinstance(data, dict):
            logging.warning(f"Update settings attempt with non-dictionary payload: {type(data)}")
            return jsonify({'error': 'Invalid payload format: body must be a JSON object.'}), 400
            
        current_config = load_web_config()
        current_config.update(data) 
        save_web_config(current_config)
        return jsonify({'message': 'Settings updated successfully'})
    except Exception as e: # Catch broader exceptions during update
        logging.error(f"Error processing update_settings request: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

def run_evolution_thread(evolution_id, initial_program_content, evaluator_script_content, config_overrides, base_web_config):
    """
    Target function for the evolution thread. This function handles the setup,
    execution, and result processing of a single evolution run.
    It's run in a separate thread to avoid blocking the web server.
    """
    evolutions[evolution_id]['status'] = STATUS_PREPARING
    
    # Define project root for subprocess execution context.
    # This ensures that openevolve-run.py is found and executed correctly.
    project_root_for_subprocess = PROJECT_ROOT # PROJECT_ROOT is already absolute

    # --- Temporary Directory Setup ---
    # Create a unique temporary directory for this evolution run.
    # For enhanced security, especially if the parent 'temp_evolution_files' directory
    # has broader permissions, consider using `tempfile.mkdtemp()` for `temp_dir_base`.
    # However, for this application, direct creation is acceptable if parent dir is controlled.
    temp_dir_base = get_full_path(TEMP_EVOLUTION_DIR_NAME)
    os.makedirs(temp_dir_base, exist_ok=True) # Ensure parent temp dir exists
    
    current_run_temp_dir = os.path.join(temp_dir_base, evolution_id)
    os.makedirs(current_run_temp_dir, exist_ok=True)
    logging.info(f"Temporary directory for evolution {evolution_id}: {current_run_temp_dir}")

    # Define paths for temporary files within the unique run directory
    initial_program_path = os.path.join(current_run_temp_dir, 'initial_program.py')
    evaluator_script_path = os.path.join(current_run_temp_dir, 'evaluator_script.py')
    temp_config_path = os.path.join(current_run_temp_dir, 'run_config.yaml')

    try:
        # --- File Preparation ---
        logging.info(f"[{evolution_id}] Writing initial program to {initial_program_path}")
        with open(initial_program_path, 'w') as f:
            f.write(initial_program_content)
        
        logging.info(f"[{evolution_id}] Writing evaluator script to {evaluator_script_path}")
        with open(evaluator_script_path, 'w') as f:
            f.write(evaluator_script_content)

        # --- Configuration Preparation ---
        # Start with the current web_config and apply any overrides from the request.
        run_config = base_web_config.copy() 
        if config_overrides: # Should already be a dict from JSON parsing
            run_config.update(config_overrides)
        
        # Ensure 'output_dir' for openevolve-run.py is set, defaulting to a subdir of current_run_temp_dir.
        if 'output_dir' not in run_config:
            run_config['output_dir'] = os.path.join(current_run_temp_dir, 'evolution_output')
        # Ensure the output directory exists
        os.makedirs(run_config['output_dir'], exist_ok=True)
        logging.info(f"[{evolution_id}] Output directory set to: {run_config['output_dir']}")

        logging.info(f"[{evolution_id}] Writing run-specific config to {temp_config_path}")
        with open(temp_config_path, 'w') as f:
            yaml.dump(run_config, f, indent=4)

        # --- Subprocess Execution ---
        # Construct the command to run `openevolve-run.py`.
        # It's assumed that `openevolve-run.py` is in the project root.
        openevolve_script_path = os.path.join(project_root_for_subprocess, 'openevolve-run.py')
        command = [
            'python', # Assuming 'python' is in PATH and points to the correct interpreter
            openevolve_script_path,
            '--config_file', temp_config_path,
            '--initial_program_file', initial_program_path,
            '--evaluator_script_file', evaluator_script_path
            # Future: Add other necessary arguments for openevolve-run.py if any
        ]
        
        evolutions[evolution_id]['status'] = STATUS_RUNNING
        logging.info(f"[{evolution_id}] Executing command: {' '.join(command)}")
        logging.info(f"[{evolution_id}] Subprocess CWD: {project_root_for_subprocess}")

        # Run the evolution process using subprocess.Popen
        process = subprocess.Popen(
            command, 
            cwd=project_root_for_subprocess, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, # Ensures stdout/stderr are strings
            bufsize=1, # Line buffered
            universal_newlines=True # For text=True
        )
        
        # Capture stdout and stderr
        # For long-running processes, consider process.stdout.readline() in a loop
        # to stream output for real-time updates if the UI supports it.
        # For now, communicate() waits for completion.
        stdout, stderr = process.communicate() 

        evolutions[evolution_id]['result'] = {'stdout': stdout, 'stderr': stderr} # Store all output

        if process.returncode == 0:
            evolutions[evolution_id]['status'] = STATUS_COMPLETED
            evolutions[evolution_id]['result']['message'] = 'Evolution completed successfully.'
            evolutions[evolution_id]['result']['output_dir'] = run_config['output_dir']
            logging.info(f"[{evolution_id}] Evolution completed successfully.")
            if stderr: # Log stderr even on success, as it might contain warnings
                 logging.warning(f"[{evolution_id}] Evolution process had stderr output (even on success):\n{stderr}")
        else:
            evolutions[evolution_id]['status'] = STATUS_FAILED
            evolutions[evolution_id]['error'] = stderr or stdout or "Evolution script failed with no specific output."
            logging.error(f"[{evolution_id}] Evolution failed. Return code: {process.returncode}\nStdout:\n{stdout}\nStderr:\n{stderr}")

    except FileNotFoundError as e: # e.g. if 'python' or 'openevolve-run.py' is not found
        logging.error(f"[{evolution_id}] FileNotFoundError in evolution thread: {e}")
        evolutions[evolution_id]['status'] = STATUS_FAILED
        evolutions[evolution_id]['error'] = f"File not found error: {str(e)}. Ensure Python and required scripts are accessible."
    except yaml.YAMLError as e: # Error writing the temp config
        logging.error(f"[{evolution_id}] YAMLError in evolution thread: {e}")
        evolutions[evolution_id]['status'] = STATUS_FAILED
        evolutions[evolution_id]['error'] = f"YAML configuration error during run setup: {str(e)}"
    except IOError as e: # Error writing files
        logging.error(f"[{evolution_id}] IOError in evolution thread: {e}")
        evolutions[evolution_id]['status'] = STATUS_FAILED
        evolutions[evolution_id]['error'] = f"File input/output error: {str(e)}"
    except Exception as e: # Catch-all for other unexpected errors
        logging.error(f"[{evolution_id}] Unexpected error in evolution thread: {e}", exc_info=True) # Log full traceback
        evolutions[evolution_id]['status'] = STATUS_FAILED
        evolutions[evolution_id]['error'] = f"An unexpected error occurred: {str(e)}"
    # 'finally' block for cleanup is currently omitted as temp files are kept for inspection.
    # If cleanup is desired:
    # finally:
    #     if os.path.exists(current_run_temp_dir):
    #         import shutil
    #         shutil.rmtree(current_run_temp_dir)
    #         logging.info(f"[{evolution_id}] Cleaned up temporary directory: {current_run_temp_dir}")


@app.route('/api/run_evolution', methods=['POST'])
def run_evolution_endpoint(): # Renamed
    """
    API endpoint to start a new evolution run.
    Expects JSON payload with 'initial_program', 'evaluator_script', and optional 'config_overrides'.
    """
    try:
        data = request.get_json()
        if not data:
            logging.warning("Run evolution attempt with invalid/empty JSON payload.")
            return jsonify({'error': 'Invalid JSON payload. Ensure Content-Type is application/json.'}), 400

        initial_program_content = data.get('initial_program')
        evaluator_script_content = data.get('evaluator_script')
        # Ensure config_overrides is a dict, even if not provided or null in JSON
        # config_overrides = data.get('config_overrides') if isinstance(data.get('config_overrides'), dict) else {}
        # Corrected handling for config_overrides YAML string:
        config_overrides_yaml_str = data.get('config_overrides', '')
        config_overrides = {} # Default to empty dict

        if isinstance(config_overrides_yaml_str, str) and config_overrides_yaml_str.strip():
            try:
                parsed_overrides = yaml.safe_load(config_overrides_yaml_str)
                if isinstance(parsed_overrides, dict):
                    config_overrides = parsed_overrides
                else:
                    logging.warning(f"Parsed config_overrides YAML is not a dictionary. Content: '{config_overrides_yaml_str}'. Ignoring.")
                    # Optionally, return a 400 error here if malformed YAML should be rejected
                    # return jsonify({'error': 'config_overrides must be a valid YAML dictionary.'}), 400
            except yaml.YAMLError as e:
                logging.warning(f"Error parsing config_overrides YAML: {e}. Content: '{config_overrides_yaml_str}'. Ignoring.")
                # Optionally, return a 400 error here
                # return jsonify({'error': f'Invalid YAML in config_overrides: {str(e)}'}), 400
        elif isinstance(config_overrides_yaml_str, dict): # If client sends it pre-parsed (e.g. from JSON)
            config_overrides = config_overrides_yaml_str


        if not initial_program_content or not evaluator_script_content:
            logging.warning("Run evolution attempt missing initial_program or evaluator_script.")
            return jsonify({'error': 'Missing initial_program or evaluator_script content in payload.'}), 400
        
        # Validate content types (simple check, can be more robust)
        if not isinstance(initial_program_content, str) or not isinstance(evaluator_script_content, str):
            return jsonify({'error': 'initial_program and evaluator_script must be strings.'}), 400


        evolution_id = uuid.uuid4().hex
        evolutions[evolution_id] = {
            'status': STATUS_STARTING, 
            'progress': 0, # Progress tracking can be implemented by openevolve-run.py if it updates a shared status file/db
            'result': None, 
            'error': None,
            'config_overrides': config_overrides # Store for reference
        }

        base_web_config = load_web_config() # Load current web config to be used as base for the run

        # Start the evolution process in a new thread
        thread = threading.Thread(target=run_evolution_thread, args=(
            evolution_id, 
            initial_program_content, 
            evaluator_script_content, 
            config_overrides,
            base_web_config
        ))
        thread.daemon = True # Allows main program to exit even if threads are running
        thread.start()
        logging.info(f"Started evolution run with ID: {evolution_id}")

        return jsonify({'message': 'Evolution started successfully.', 'evolution_id': evolution_id}), 202 # 202 Accepted
    except Exception as e:
        logging.error(f"Error processing run_evolution request: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


@app.route('/api/evolution_status/<string:evolution_id>', methods=['GET'])
def evolution_status_endpoint(evolution_id): # Renamed
    """API endpoint to get the status of a specific evolution run."""
    status_info = evolutions.get(evolution_id)
    if not status_info:
        logging.warning(f"Status requested for non-existent evolution ID: {evolution_id}")
        return jsonify({'status': STATUS_NOT_FOUND, 'error': 'Evolution ID not found.'}), 404
    return jsonify(status_info)

@app.route('/api/evolution_results/<string:evolution_id>', methods=['GET'])
def evolution_results_endpoint(evolution_id): # Renamed
    """API endpoint to get the results of a completed evolution run."""
    evolution_data = evolutions.get(evolution_id)
    if not evolution_data:
        logging.warning(f"Results requested for non-existent evolution ID: {evolution_id}")
        return jsonify({'status': STATUS_NOT_FOUND, 'error': 'Evolution ID not found.'}), 404

    # Return current data even if not completed, client can interpret based on status
    return jsonify(evolution_data)


if __name__ == '__main__':
    # Note: Flask's default development server is not suitable for production.
    # Use a production-ready WSGI server (e.g., Gunicorn, uWSGI) for deployment.
    logging.info(f"Starting Flask app. PROJECT_ROOT is {PROJECT_ROOT}")
    app.run(debug=True, host='0.0.0.0', port=5001) # Changed port for clarity if other apps run on 5000
```
