(function() { // IIFE to encapsulate the entire script
'use strict'; // Enable strict mode

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    // Tabs & Content
    const playgroundTab = document.getElementById('playground-tab');
    const settingsTab = document.getElementById('settings-tab');
    const playgroundContent = document.getElementById('playground-content');
    const settingsContent = document.getElementById('settings-content');

    // Theme Toggle Elements
    const themeToggleButton = document.getElementById('theme-toggle');

    // Playground Tab Elements
    const initialProgramTextarea = document.getElementById('initial-program');
    const evaluatorScriptTextarea = document.getElementById('evaluator-script');
    const configOverridesTextarea = document.getElementById('config-overrides');
    const runEvolutionButton = document.getElementById('run-evolution-btn');
    const evolutionStatusText = document.getElementById('evolution-status-text'); // This is the span/div for messages
    const evolutionProgressBar = document.getElementById('evolution-progress');
    const evolutionResultsDisplay = document.getElementById('evolution-results-display');
    const bestProgramCodeElement = document.getElementById('best-program-code');

    // Settings Tab Elements
    const settingsFormContainer = document.getElementById('settings-form-container');
    const settingsMessage = document.getElementById('settings-message'); // For settings save status
    const saveSettingsButton = document.getElementById('save-settings-btn'); 

    // --- Global State Variables ---
    let currentEvolutionId = null; // Stores the ID of the currently active evolution run
    let pollTimeoutId = null;    // Stores the timeout ID for polling evolution status

    // --- Constants ---
    const POLLING_INTERVAL = 3000; // Milliseconds for polling status
    const THEME_LIGHT = 'light';
    const THEME_DARK = 'dark';

    // --- Utility Functions ---

    /**
     * Formats a settings key (e.g., "llm_model_name") into a readable label (e.g., "Llm Model Name").
     * @param {string} key - The key to format.
     * @returns {string} The formatted label.
     */
    function formatLabel(key) {
        if (typeof key !== 'string') return '';
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    /**
     * Sets a message in a designated element, with a specific class for styling.
     * @param {HTMLElement} element - The DOM element to display the message in.
     * @param {string} message - The message text.
     * @param {'info' | 'success' | 'error'} type - The type of message (for CSS class).
     */
    function showUserMessage(element, message, type) {
        if (!element) {
            console.warn("Attempted to show message on a non-existent element:", message);
            return;
        }
        element.textContent = message;
        // Assumes CSS classes like .message-info, .message-success, .message-error are defined
        // For evolutionStatusText, it might also use .status-info, .status-error, etc.
        element.className = `status-message message-${type}`; 
    }


    // --- Tab Switching Logic ---

    /**
     * Activates a specific tab and its content panel, deactivating others.
     * @param {HTMLElement} tabToActivate - The tab button element to activate.
     * @param {HTMLElement} contentToDisplay - The content panel element to display.
     * @param {Array<HTMLElement>} allTabs - An array of all tab button elements.
     * @param {Array<HTMLElement>} allContents - An array of all tab content panel elements.
     */
    function activateTab(tabToActivate, contentToDisplay, allTabs, allContents) {
        allTabs.forEach(t => t.classList.remove('active'));
        allContents.forEach(c => c.classList.add('hidden'));

        if (tabToActivate) tabToActivate.classList.add('active');
        if (contentToDisplay) contentToDisplay.classList.remove('hidden');

        // Specific actions when a tab is activated
        if (tabToActivate === settingsTab) {
            loadSettings(); // Load settings when the Settings tab is activated
        }
        // Add other tab-specific actions here if needed
    }

    if (playgroundTab && settingsTab && playgroundContent && settingsContent) {
        const mainTabsArray = [playgroundTab, settingsTab];
        const mainContentsArray = [playgroundContent, settingsContent];

        mainTabsArray.forEach((tabButton, index) => {
            if (tabButton) { // Ensure tab element exists
                tabButton.addEventListener('click', () => {
                    activateTab(tabButton, mainContentsArray[index], mainTabsArray, mainContentsArray);
                });
            } else {
                console.error(`Tab button at index ${index} not found.`);
            }
        });

        // Initialize: Show Playground tab by default if it exists
        if (playgroundTab && playgroundContent) {
            activateTab(playgroundTab, playgroundContent, mainTabsArray, mainContentsArray);
        } else if (settingsTab && settingsContent) { // Fallback if playground tab is missing
             activateTab(settingsTab, settingsContent, mainTabsArray, mainContentsArray);
        } else {
            console.error("Default tab (Playground) or its content not found. No tab will be active by default.");
        }
    } else {
        console.error("Core tab elements (playgroundTab, settingsTab, etc.) not found! Tab functionality will be impaired.");
    }

    // --- Theme Switching Logic ---
    /**
     * Applies the specified theme to the document and saves it to localStorage.
     * @param {string} themeName - The name of the theme to apply (e.g., 'light' or 'dark').
     */
    function applyTheme(themeName) {
        document.body.setAttribute('data-theme', themeName);
        localStorage.setItem('theme', themeName);
    }

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            const newTheme = document.body.getAttribute('data-theme') === THEME_DARK ? THEME_LIGHT : THEME_DARK;
            applyTheme(newTheme);
        });
    } else {
        console.error("Theme toggle button (#theme-toggle) not found!");
    }
    // Load saved theme on initial page load, defaulting to light theme
    const savedUserTheme = localStorage.getItem('theme') || THEME_LIGHT;
    applyTheme(savedUserTheme);


    // --- Playground Tab Functionality ---

    /**
     * Clears any active polling timeout for evolution status.
     */
    function clearPolling() {
        if (pollTimeoutId) {
            clearTimeout(pollTimeoutId);
            pollTimeoutId = null;
        }
    }

    /**
     * Fetches and displays the results of a completed evolution.
     * @param {string} evolutionId - The ID of the evolution.
     */
    async function fetchResults(evolutionId) {
        if (!evolutionResultsDisplay) {
            console.error("Evolution results display element not found.");
            return;
        }
        evolutionResultsDisplay.innerHTML = ''; // Clear previous results

        try {
            const response = await fetch(`/api/evolution_results/${evolutionId}`);
            const results = await response.json(); 

            if (!response.ok) {
                const errorMsg = results?.error || response.statusText;
                const p = document.createElement('p');
                p.className = 'error-message';
                p.textContent = `Error fetching results (HTTP ${response.status}): ${errorMsg}`;
                evolutionResultsDisplay.appendChild(p);
                return;
            }
            
            const heading = document.createElement('h3');
            heading.textContent = 'Evolution Results:';
            evolutionResultsDisplay.appendChild(heading);

            if (results.result) { 
                if (results.result.output_dir) {
                    const p = document.createElement('p');
                    const strong = document.createElement('strong');
                    strong.textContent = 'Output Directory: ';
                    p.appendChild(strong);
                    p.appendChild(document.createTextNode(results.result.output_dir));
                    evolutionResultsDisplay.appendChild(p);
                }
                if (results.result.stdout) {
                    const p = document.createElement('p');
                    const strong = document.createElement('strong');
                    strong.textContent = 'Standard Output:';
                    p.appendChild(strong);
                    evolutionResultsDisplay.appendChild(p);
                    const pre = document.createElement('pre');
                    pre.textContent = results.result.stdout; // Safely set stdout
                    evolutionResultsDisplay.appendChild(pre);
                }
                if (results.result.stderr) {
                    const p = document.createElement('p');
                    const strong = document.createElement('strong');
                    strong.textContent = 'Standard Error:';
                    p.appendChild(strong);
                    evolutionResultsDisplay.appendChild(p);
                    const pre = document.createElement('pre');
                    pre.textContent = results.result.stderr; // Safely set stderr
                    evolutionResultsDisplay.appendChild(pre);
                }
                if (bestProgramCodeElement) {
                     bestProgramCodeElement.textContent = results.result.best_program_content || '// Best program content not directly available via this API view.';
                }
            } else {
                 const p = document.createElement('p');
                 p.textContent = 'No detailed result object was returned by the server.';
                 evolutionResultsDisplay.appendChild(p);
            }
            
            const noSpecificResults = !results.result || 
                                     (Object.keys(results.result).length === 0 && !results.result?.best_program_content);
            // Avoid duplicate "no specific results" if stdout/stderr were already shown (even if empty)
            if (noSpecificResults && !(results.result?.stdout || results.result?.stderr || results.result?.output_dir)) {
                 const p = document.createElement('p');
                 p.textContent = 'No specific result data was found for this evolution.';
                 evolutionResultsDisplay.appendChild(p);
            }

        } catch (error) { 
            console.error('Error fetching results:', error);
            const p = document.createElement('p');
            p.className = 'error-message';
            p.textContent = `An application error occurred while fetching results: ${error.message}`;
            evolutionResultsDisplay.appendChild(p);
        }
    }

    /**
     * Polls the server for the status of an ongoing evolution.
     * Updates UI elements (status text, progress bar) accordingly.
     * @param {string} evolutionId - The ID of the evolution to poll.
     */
    async function pollStatus(evolutionId) {
        clearPolling(); 

        if (!evolutionStatusText || !evolutionProgressBar || !runEvolutionButton || !evolutionResultsDisplay) {
            console.error("One or more UI elements for polling are missing. Aborting poll.");
            return;
        }

        try {
            const response = await fetch(`/api/evolution_status/${evolutionId}`);
            const statusData = await response.json(); 

            if (!response.ok) {
                const errorMessage = `Error fetching status (HTTP ${response.status}): ${statusData?.error || response.statusText}`;
                showUserMessage(evolutionStatusText, errorMessage, 'error');
                evolutionProgressBar.style.visibility = 'hidden';
                runEvolutionButton.disabled = false; 
                return;
            }
            
            showUserMessage(evolutionStatusText, `Status: ${statusData.status || 'Unknown'}`, 'info');
             // Apply specific status class for targeted CSS if needed (e.g. .status-running, .status-completed)
            if(evolutionStatusText) evolutionStatusText.className = `status-message message-info status-${statusData.status || 'unknown'}`;


            // Update progress bar
            if (statusData.progress !== undefined && statusData.progress !== null) {
                evolutionProgressBar.value = statusData.progress;
                evolutionProgressBar.style.visibility = 'visible';
            } else if (['running', 'preparing', 'starting'].includes(statusData.status)) {
                evolutionProgressBar.removeAttribute('value'); // Indeterminate state
                evolutionProgressBar.style.visibility = 'visible';
            } else {
                evolutionProgressBar.style.visibility = 'hidden';
            }
            
            // Handle terminal states
            if (statusData.status === 'completed') {
                evolutionProgressBar.value = 100;
                showUserMessage(evolutionStatusText, 'Evolution completed successfully!', 'success');
                fetchResults(evolutionId);
                runEvolutionButton.disabled = false;
            } else if (statusData.status === 'failed') {
                let errorMsg = `Evolution failed. ${statusData.error || 'No specific error message.'}`;
                let detailStdErr = statusData.result?.stderr;
                let detailStdOut = (statusData.result?.stdout && !statusData.result?.stderr) ? statusData.result.stdout : null;
                
                // Display the primary, potentially generic, error message in the status line
                showUserMessage(evolutionStatusText, errorMsg, 'error'); 
                
                if (evolutionResultsDisplay) { 
                    evolutionResultsDisplay.innerHTML = ''; // Clear previous content
                    const pError = document.createElement('p');
                    pError.className = 'error-message';
                    const strong = document.createElement('strong');
                    strong.textContent = 'Error during evolution:';
                    pError.appendChild(strong);
                    pError.appendChild(document.createElement('br'));
                    pError.appendChild(document.createTextNode(errorMsg)); // Main error text
                    evolutionResultsDisplay.appendChild(pError);

                    // Safely display stderr if present
                    if(detailStdErr) {
                        const pStderrLabel = document.createElement('p');
                        const strongStderr = document.createElement('strong');
                        strongStderr.textContent = 'Standard Error:';
                        pStderrLabel.appendChild(strongStderr);
                        evolutionResultsDisplay.appendChild(pStderrLabel);
                        const preStderr = document.createElement('pre');
                        preStderr.textContent = detailStdErr;
                        evolutionResultsDisplay.appendChild(preStderr);
                    }
                    // Safely display stdout if present (and no stderr, to avoid redundancy if stderr includes stdout)
                    if(detailStdOut) {
                        const pStdoutLabel = document.createElement('p');
                        const strongStdout = document.createElement('strong');
                        strongStdout.textContent = 'Standard Output:';
                        pStdoutLabel.appendChild(strongStdout);
                        evolutionResultsDisplay.appendChild(pStdoutLabel);
                        const preStdout = document.createElement('pre');
                        preStdout.textContent = detailStdOut;
                        evolutionResultsDisplay.appendChild(preStdout);
                    }
                }
                runEvolutionButton.disabled = false;
            } else if (['running', 'preparing', 'starting'].includes(statusData.status)) {
                // Continue polling for active states
                pollTimeoutId = setTimeout(() => pollStatus(evolutionId), POLLING_INTERVAL);
            } else { // Unhandled or unknown status
                showUserMessage(evolutionStatusText, evolutionStatusText.textContent + ' - Polling stopped.', 'info');
                runEvolutionButton.disabled = false;
                evolutionProgressBar.style.visibility = 'hidden';
            }

        } catch (error) { 
            console.error('Polling error:', error);
            showUserMessage(evolutionStatusText, `Polling error: ${error.message}. Check console.`, 'error');
            evolutionProgressBar.style.visibility = 'hidden';
            runEvolutionButton.disabled = false;
            clearPolling();
        }
    }

    if (runEvolutionButton) {
        runEvolutionButton.addEventListener('click', async () => {
            if (!initialProgramTextarea || !evaluatorScriptTextarea || !configOverridesTextarea || 
                !evolutionStatusText || !evolutionProgressBar || !evolutionResultsDisplay || !bestProgramCodeElement) {
                console.error("Playground UI elements missing. Cannot run evolution.");
                alert("Error: Critical UI elements are missing. Cannot proceed.");
                return;
            }

            const initialProgramCode = initialProgramTextarea.value.trim();
            const evaluatorScriptCode = evaluatorScriptTextarea.value.trim();
            const configOverridesYAML = configOverridesTextarea.value; 

            if (!initialProgramCode) {
                alert('Initial Program Code cannot be empty.');
                initialProgramTextarea.focus();
                return;
            }
            if (!evaluatorScriptCode) {
                alert('Evaluator Script Code cannot be empty.');
                evaluatorScriptTextarea.focus();
                return;
            }

            clearPolling();
            showUserMessage(evolutionStatusText, 'Initializing evolution...', 'info');
            evolutionProgressBar.style.visibility = 'visible';
            evolutionProgressBar.value = 0;
            evolutionResultsDisplay.innerHTML = '<p>Results will appear here once the evolution is complete.</p>';
            if(bestProgramCodeElement) bestProgramCodeElement.textContent = '';
            runEvolutionButton.disabled = true;

            const payload = {
                initial_program: initialProgramCode,
                evaluator_script: evaluatorScriptCode,
                config_overrides: configOverridesYAML 
            };

            try {
                const response = await fetch('/api/run_evolution', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload),
                });
                const data = await response.json(); 

                if (!response.ok) {
                    const serverErrorMsg = data?.error || `HTTP ${response.status}: ${response.statusText}`;
                    throw new Error(`Server error starting evolution: ${serverErrorMsg}`);
                }

                currentEvolutionId = data.evolution_id;
                showUserMessage(evolutionStatusText, `Evolution started (ID: ${currentEvolutionId}). Polling for status...`, 'info');
                pollStatus(currentEvolutionId);

            } catch (error) { 
                console.error('Error running evolution:', error);
                showUserMessage(evolutionStatusText, `Error: ${error.message}`, 'error');
                evolutionProgressBar.style.visibility = 'hidden';
                runEvolutionButton.disabled = false; 
            }
        });
    } else {
        console.error("Run Evolution button (#run-evolution-btn) not found!");
    }

    // Initial UI state for Playground
    if (evolutionProgressBar) evolutionProgressBar.style.visibility = 'hidden';
    if (evolutionStatusText) {
        showUserMessage(evolutionStatusText, 'Idle. Ready to start evolution.', 'info');
    }
    if (runEvolutionButton) runEvolutionButton.disabled = false;


    // --- Settings Tab Functionality ---

    /**
     * Creates and returns a DOM element for a single setting or a group of settings.
     * Recursively called for nested objects to create fieldsets.
     * @param {string} key - The key of the setting.
     * @param {*} value - The value of the setting.
     * @param {string} parentKeyPath - The path of parent keys (e.g., "llm.model"), used for unique IDs and names.
     * @returns {HTMLElement} The DOM element representing the setting input or group.
     */
    function createSettingInput(key, value, parentKeyPath) {
        const settingEntry = document.createElement('div');
        settingEntry.classList.add('setting-entry');
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
             settingEntry.classList.add('setting-group'); // For fieldsets
        }

        const currentPath = parentKeyPath ? `${parentKeyPath}.${key}` : key;
        const inputId = `setting-${currentPath.replace(/[._]/g, '-')}`; 

        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
            const fieldset = document.createElement('fieldset');
            const legend = document.createElement('legend');
            legend.textContent = formatLabel(key);
            legend.title = `Setting group: ${currentPath}`; 
            fieldset.appendChild(legend);

            for (const subKey in value) {
                if (Object.prototype.hasOwnProperty.call(value, subKey)) {
                    fieldset.appendChild(createSettingInput(subKey, value[subKey], currentPath));
                }
            }
            settingEntry.appendChild(fieldset);
        } else { 
            const label = document.createElement('label');
            label.textContent = formatLabel(key);
            label.title = `Setting: ${currentPath}`; 
            label.setAttribute('for', inputId);
            
            let input;
            if (typeof value === 'boolean') {
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = value;
            } else if (typeof value === 'number') {
                input = document.createElement('input');
                input.type = 'number';
                input.value = value;
                if (!Number.isInteger(value)) input.step = "any"; 
            } else { 
                input = document.createElement('input');
                input.type = 'text';
                input.value = (value === null || value === undefined) ? '' : String(value); 
            }
            input.id = inputId;
            input.name = currentPath; 
            
            settingEntry.appendChild(label);
            settingEntry.appendChild(input);
        }
        return settingEntry;
    }

    /**
     * Loads settings from the API and dynamically populates the settings form.
     */
    async function loadSettings() {
        if (!settingsFormContainer || !settingsMessage) {
            console.error("Settings form container or message element not found. Cannot load settings.");
            return;
        }

        settingsFormContainer.innerHTML = '<p>Loading settings...</p>'; 
        showUserMessage(settingsMessage, 'Loading settings...', 'info');

        try {
            const response = await fetch('/api/get_settings');
            const settings = await response.json(); 

            if (!response.ok) {
                const errorText = settings?.error || response.statusText;
                throw new Error(`Failed to load settings (HTTP ${response.status}): ${errorText}`);
            }
            
            settingsFormContainer.innerHTML = ''; 

            if (Object.keys(settings).length === 0) {
                settingsFormContainer.innerHTML = '<p>No settings were found, or the settings configuration is empty.</p>';
                showUserMessage(settingsMessage, 'No settings found.', 'info');
                return;
            }

            for (const key in settings) {
                if (Object.prototype.hasOwnProperty.call(settings, key)) {
                    settingsFormContainer.appendChild(createSettingInput(key, settings[key], ''));
                }
            }
            showUserMessage(settingsMessage, 'Settings loaded successfully.', 'success');
        } catch (error) { 
            console.error('Error loading settings:', error);
            const userErrorMessage = `An error occurred while loading settings: ${error.message}. Please check server logs or try again.`;
            settingsFormContainer.innerHTML = `<p class="error-message">${userErrorMessage}</p>`;
            showUserMessage(settingsMessage, 'Error loading settings.', 'error');
        }
    }
    
    /**
     * Reconstructs the nested settings object from the current form inputs.
     * @returns {object} The settings object derived from the form.
     */
    function getSettingsFromForm() {
        const settings = {};
        if (!settingsFormContainer) {
            console.error("Settings form container not found. Cannot get settings from form.");
            return settings; 
        }

        const inputs = settingsFormContainer.querySelectorAll('input, select, textarea');

        inputs.forEach(input => {
            const name = input.name;
            if (!name) return; 

            let value;
            if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.type === 'number') {
                const numVal = input.value.trim();
                value = numVal === '' ? null : parseFloat(numVal);
                if (isNaN(value) && numVal !== '') { 
                     console.warn(`Invalid number for input "${name}": "${numVal}". Sending original string. Consider client-side validation.`);
                     value = numVal; 
                }
            } else { 
                value = input.value;
            }

            const keys = name.split('.');
            let currentLevel = settings;
            keys.forEach((key, index) => {
                if (index === keys.length - 1) {
                    currentLevel[key] = value;
                } else {
                    if (!currentLevel[key] || typeof currentLevel[key] !== 'object') {
                        currentLevel[key] = {}; 
                    }
                    currentLevel = currentLevel[key];
                }
            });
        });
        return settings;
    }

    if (saveSettingsButton) {
        saveSettingsButton.addEventListener('click', async () => {
            if (!settingsMessage) {
                console.error("Settings message element not found. Cannot display save status.");
                return;
            }
            showUserMessage(settingsMessage, 'Saving settings...', 'info');

            const settingsData = getSettingsFromForm();

            try {
                const response = await fetch('/api/update_settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(settingsData),
                });

                const responseData = await response.json().catch(() => null); 

                if (!response.ok) {
                    let errorMsg = `Failed to save settings (HTTP ${response.status}): ${response.statusText}.`;
                    if (responseData?.error) errorMsg += ` Server: ${responseData.error}`;
                    else if (responseData && typeof responseData === 'string') errorMsg += ` Server: ${responseData}`;
                    else if (response.text) { 
                        const textError = await response.text().catch(() => '');
                        if(textError) errorMsg += ` Server: ${textError}`;
                    }
                    throw new Error(errorMsg);
                }
                
                showUserMessage(settingsMessage, responseData?.message || 'Settings saved successfully!', 'success');
                
            } catch (error) { 
                console.error('Error saving settings:', error);
                showUserMessage(settingsMessage, `Error: ${error.message}`, 'error');
            }
        });
    } else {
        console.error("Save Settings button (#save-settings-btn) not found!");
    }

    console.log("OpenEvolve script fully loaded and initialized.");
});

})(); // End of IIFE
