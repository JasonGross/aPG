// static/script.js

let allEssays = [];
let responseCache = {};

document.addEventListener('DOMContentLoaded', () => {
    const promptForm = document.getElementById('prompt-form');
    const promptInput = document.getElementById('prompt-input');
    const submitButton = document.getElementById('submit-button');
    const responseOutput = document.getElementById('response-output');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');
    const essaysList = document.getElementById('essays-list');
    const sortButtons = document.querySelectorAll('.sort-button');
    const charCount = document.getElementById('char-count');
    const thinkingCheckbox = document.getElementById('thinking-checkbox'); // Get the checkbox

    let currentSort = { field: 'time', order: 'desc' }; // Default sort
    let eventSource = null; // To hold the EventSource connection
    let titleRendered = false; // Flag to track if the title part has been rendered

    // --- Helper: Finalize UI State ---
    function finalizeUI(success = true) {
        promptInput.disabled = false;
        submitButton.disabled = false;
        submitButton.textContent = 'Generate Essay';
        loadingIndicator.classList.add('hidden');
        if (success) {
            fetchEssays(); // Refresh essay list on success
        }
        console.log("UI finalized.");
    }

    // --- Helper: Generate Cache Key ---
    function getCacheKey(prompt, thinkingEnabled) {
        return `${prompt}|thinking=${thinkingEnabled}`;
    }

    // --- Helper: Render Raw Text to Formatted HTML ---
    function renderFormattedText(rawText) {
        if (!rawText) return '';

        // Escape HTML
        const escapedText = rawText
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');

        // Find first paragraph break for title
        const firstParaBreakIndex = escapedText.search(/\n+/);
        let htmlOutput = "";

        if (firstParaBreakIndex !== -1) {
            const title = escapedText.substring(0, firstParaBreakIndex);
            const restOfText = escapedText.substring(firstParaBreakIndex);
            // Format: Title span + replaced newlines in the rest
            htmlOutput = `<span class="essay-title">${title}</span>${restOfText.replace(/\n+/g, '<br><br>')}`;
        } else {
            // If no newline sequence, treat the whole text as the title
            htmlOutput = `<span class="essay-title">${escapedText}</span>`;
        }
        return htmlOutput;
    }

    // --- Character Counter ---
    promptInput.addEventListener('input', () => {
        const count = promptInput.value.length;
        charCount.textContent = `${count} / 70 characters`;
    });

    // --- Render Essays (Handles Sorting and DOM Update) ---
    function renderEssays(essays) {
        console.log(`Rendering essays. Current sort: ${currentSort.field} ${currentSort.order}`);
        essaysList.innerHTML = ''; // Clear list

        if (!Array.isArray(essays)) {
            console.error("Invalid data provided to renderEssays. Expected an array.", essays);
            essaysList.innerHTML = '<li class="text-red-500">Error displaying essays.</li>';
            return;
        }

        if (essays.length === 0) {
            essaysList.innerHTML = '<li class="text-gray-500">No essays found yet.</li>';
            return;
        }

        // --- Client-Side Sorting --- //
        const sortedEssays = [...essays].sort((a, b) => {
            let valA, valB;

            switch (currentSort.field) {
                case 'alpha':
                    valA = a.prompt?.toLowerCase() || '';
                    valB = b.prompt?.toLowerCase() || '';
                    break;
                case 'views':
                    valA = a.view_count ?? 0;
                    valB = b.view_count ?? 0;
                    break;
                case 'time':
                default:
                    // Handle potentially null or invalid date strings
                    valA = a.created_at ? new Date(a.created_at).getTime() : 0;
                    valB = b.created_at ? new Date(b.created_at).getTime() : 0;
                    if (isNaN(valA)) valA = 0; // Fallback for invalid dates
                    if (isNaN(valB)) valB = 0;
                    break;
            }

            let comparison = 0;
            if (valA > valB) {
                comparison = 1;
            } else if (valA < valB) {
                comparison = -1;
            }

            return currentSort.order === 'desc' ? (comparison * -1) : comparison;
        });
        // ------------------------- //

        console.log("Rendering sorted essays...");
        sortedEssays.forEach((essay, index) => {
            console.log(`Rendering essay ${index + 1}:`, essay);
            const li = document.createElement('li');
            const promptText = essay.prompt ? essay.prompt : '[No prompt text]';
            const viewCount = essay.view_count ?? 0; // Get view count, default to 0

            // Create span for prompt text
            const promptSpan = document.createElement('span');
            promptSpan.textContent = promptText;

            // Create span for view count (initially hidden)
            const viewCountSpan = document.createElement('span');
            viewCountSpan.classList.add('view-count-display', 'ml-2', 'text-gray-500'); // Add styling
            viewCountSpan.textContent = `(Views: ${viewCount})`;
            viewCountSpan.style.display = 'none'; // Hide by default

            // Append spans to list item
            li.appendChild(promptSpan);
            li.appendChild(viewCountSpan);

            li.classList.add('cursor-pointer', 'hover:text-orange-700');

            // Event listeners for hover effect
            li.addEventListener('mouseenter', () => {
                if (currentSort.field === 'views') {
                    viewCountSpan.style.display = 'inline'; // Show on hover if sorting by views
                }
            });

            li.addEventListener('mouseleave', () => {
                viewCountSpan.style.display = 'none'; // Always hide when mouse leaves
            });

            // --- Modified Click Listener ---
            li.addEventListener('click', () => {
                const clickedPrompt = essay.prompt;
                console.log("Essay clicked:", clickedPrompt);

                // Update URL without reload
                try {
                    const url = new URL(window.location);
                    url.searchParams.set('prompt', clickedPrompt); // Set/update the prompt parameter
                    // Use pushState to change URL without full page load
                    history.pushState({ prompt: clickedPrompt }, '', url.toString());
                    console.log("URL updated to:", url.toString());
                } catch (e) {
                    console.error("Error updating URL:", e);
                    // Fallback or simply proceed without URL change if needed
                }

                // Set the input value and simulate submission (existing logic)
                promptInput.value = clickedPrompt;
                promptInput.dispatchEvent(new Event('input')); // Update char count
                promptForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
            });
            // ----------------------------- //

            essaysList.appendChild(li);
            console.log(`Appended item ${index + 1} to the list.`);
        });
        console.log("Finished rendering essays.");
    }

    // --- Fetch Essays (Only Fetches, Does Not Sort/Render) ---
    async function fetchEssays() {
        console.log(`Fetching all essays from backend...`);

        // Show loading state ONLY if the list is currently empty of actual essay items
        if (!essaysList.querySelector('li')) { // Check if any <li> elements exist
            console.log("Essay list is empty, showing loading message.");
            essaysList.innerHTML = '<li class="text-gray-400">Loading essays...</li>';
        } else {
            console.log("Essay list has items, keeping them visible during fetch.");
            // Keep existing list displayed during refresh
        }

        try {
            // Fetch from the deployed worker URL
            const response = await fetch(`https://apg-worker.jasongross9.workers.dev/essays`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const essays = await response.json();
            console.log("Fetched essays data:", essays);

            if (!Array.isArray(essays)) {
                console.error("Invalid data received from /essays. Expected an array.", essays);
                throw new Error("Received invalid data from server.");
            }

            allEssays = essays; // Store fetched essays
            renderEssays(allEssays); // Render the list with current sort

        } catch (error) {
            console.error('Error fetching essays:', error);
            essaysList.innerHTML = '<li class="text-red-500">Failed to load essays.</li>';
            allEssays = []; // Clear local store on error
        }
    }

    // --- Handle Form Submission ---
    promptForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const prompt = promptInput.value.trim();
        const thinkingEnabled = thinkingCheckbox.checked;
        let rawFullResponseAccumulator = "";

        if (!prompt) return;

        // --- Update URL with prompt and thinking state ---
        try {
            const url = new URL(window.location);
            url.searchParams.set('prompt', prompt);
            url.searchParams.set('thinking', thinkingEnabled.toString()); // Add thinking state
            history.pushState({ prompt: prompt, thinking: thinkingEnabled }, '', url.toString());
            console.log("URL updated with prompt and thinking state:", url.toString());
        } catch (e) {
            console.error("Error updating URL on submit:", e);
        }
        // ---------------------------------------------- //

        // --- Check Cache (using thinking state in key) ---
        const cacheKey = getCacheKey(prompt, thinkingEnabled);
        if (responseCache[cacheKey]) {
            console.log("Loading response from cache for key:", cacheKey);
            const cachedData = responseCache[cacheKey];
            const cachedRawText = typeof cachedData === 'string' ? cachedData : cachedData.text;
            const cachedMetadata = typeof cachedData === 'object' ? cachedData.metadata : null;

            if (cachedMetadata) {
                console.log("Using cached metadata:", cachedMetadata);
            } else {
                console.log("No metadata found in cache for this entry.");
            }

            // Basic UI update for loading
            submitButton.textContent = 'Loading Cached...';
            loadingIndicator.classList.remove('hidden');
            responseOutput.innerHTML = ''; // Clear previous output
            errorMessage.classList.add('hidden');
            promptInput.disabled = true;
            submitButton.disabled = true;

            // Use a timeout to allow UI to update before rendering
            setTimeout(() => {
                responseOutput.innerHTML = renderFormattedText(cachedRawText);
                finalizeUI(false);
                console.log("Finished displaying cached response.");
            }, 50); // Short delay

            return; // Stop execution, don't fetch from backend
        }
        // --------------- //

        console.log(`Fetching response from backend for: ${prompt}, Thinking: ${thinkingEnabled}`);
        // Reset title rendered flag for new request
        titleRendered = false;
        // Basic UI update for loading
        promptInput.disabled = true;
        submitButton.disabled = true;
        submitButton.textContent = 'Generating...';
        loadingIndicator.classList.remove('hidden');
        responseOutput.innerHTML = '';
        errorMessage.classList.add('hidden');

        // Close any existing EventSource connection (moved down)
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        try {
            // Fetch logic (POST request to deployed worker URL)
            const response = await fetch('https://apg-worker.jasongross9.workers.dev/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                // --- Send prompt and thinking state to backend --- //
                body: `prompt=${encodeURIComponent(prompt)}&thinking_enabled=${thinkingEnabled}`
            });

            if (!response.ok) {
                // Try to read error message from backend if available
                let errorData = { message: `HTTP error! status: ${response.status}` };
                try {
                    errorData = await response.json();
                } catch (e) { /* Ignore if response is not JSON */ }
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }

            if (response.headers.get('content-type')?.includes('text/event-stream')) {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let receivedMetadata = null; // Variable to store metadata from stream

                reader.read().then(function processText({ done, value }) {
                    if (done) {
                        console.log("Stream complete.");
                        // Cache the RAW response and metadata on successful completion
                        if (!errorMessage.textContent) { // Check if an error message was displayed during stream
                            responseCache[cacheKey] = { text: rawFullResponseAccumulator, metadata: receivedMetadata || {} }; // Use cacheKey
                            console.log("Response and metadata cached for key:", cacheKey);
                            finalizeUI(true); // Finalize UI (will fetch essays)
                        } else {
                            finalizeUI(false); // Finalize UI but don't refresh list if errors occurred
                        }
                        return;
                    }

                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split('\n');
                    lines.forEach(line => {
                        if (line.startsWith('data:')) {
                            try {
                                const data = JSON.parse(line.substring(5).trim());
                                if (data.metadata) { // --- Handle metadata event ---
                                    console.log("Received metadata:", data.metadata);
                                    receivedMetadata = data.metadata; // Store it
                                } else if (data.text) { // --- Handle text event ---
                                    // Accumulate RAW text for cache
                                    rawFullResponseAccumulator += data.text;

                                    // Escape the current chunk
                                    const escapedChunk = data.text
                                        .replace(/&/g, '&amp;')
                                        .replace(/</g, '&lt;')
                                        .replace(/>/g, '&gt;')
                                        .replace(/"/g, '&quot;')
                                        .replace(/'/g, '&#039;');

                                    let formattedChunkHTML = "";

                                    // --- New Title Handling Logic ---
                                    if (!titleRendered) {
                                        const firstBreakIndex = escapedChunk.search(/\n+/);

                                        if (firstBreakIndex !== -1) {
                                            // Newline found in this chunk - Title ends here
                                            const titlePart = escapedChunk.substring(0, firstBreakIndex);
                                            const restPart = escapedChunk.substring(firstBreakIndex);
                                            // Format title part + rest part (with its newlines converted)
                                            formattedChunkHTML = `<span class="essay-title">${titlePart}</span>${restPart.replace(/\n+/g, '<br><br>')}`;
                                            titleRendered = true; // Mark title as fully rendered
                                        } else {
                                            // No newline yet - entire chunk is part of the title
                                            formattedChunkHTML = `<span class="essay-title">${escapedChunk}</span>`;
                                            // titleRendered remains false
                                        }
                                    } else {
                                        // Title already rendered, process chunk normally
                                        formattedChunkHTML = escapedChunk.replace(/\n+/g, '<br><br>');
                                    }
                                    // --------------------------------- //

                                    responseOutput.innerHTML += formattedChunkHTML;
                                    // Consolidate multiple <br> tags added by chunking/newlines
                                    responseOutput.innerHTML = responseOutput.innerHTML.replace(/(<br\s*\/?><br\s*\/?>)+/g, '<br><br>');

                                } else if (data.error) {
                                    console.error("SSE Error:", data.error);
                                    errorMessage.textContent = `Error: ${data.error}`;
                                    errorMessage.classList.remove('hidden');
                                    finalizeUI(false); // Finalize UI on stream error
                                    reader.cancel(); // Stop reading on error
                                } else if (data.end) {
                                    console.log("SSE Stream ended by server.");
                                    // Cache the RAW response and metadata on server end signal (if no prior error)
                                    if (!errorMessage.textContent) {
                                        responseCache[cacheKey] = { text: rawFullResponseAccumulator, metadata: receivedMetadata || {} }; // Use cacheKey
                                        console.log("Response and metadata cached for key:", cacheKey);
                                        finalizeUI(true); // Finalize UI
                                    } else {
                                        finalizeUI(false);
                                    }
                                    reader.cancel();
                                }
                            } catch (e) {
                                console.error("Error parsing SSE data:", e, "Line:", line);
                                errorMessage.textContent = "Error processing response stream.";
                                errorMessage.classList.remove('hidden');
                                finalizeUI(false); // Finalize on parsing error
                                reader.cancel();
                            }
                        }
                    });
                    // Continue reading
                    reader.read().then(processText);
                }).catch(error => {
                    console.error("Stream reading error:", error);
                    errorMessage.textContent = `Stream error: ${error.message}`;
                    errorMessage.classList.remove('hidden');
                    finalizeUI(false); // Finalize on stream read error
                });
            } else {
                // Handle non-streaming response
                const rawText = await response.text();
                // Cache raw text in the new object format, but without metadata
                responseCache[cacheKey] = { text: rawText, metadata: null }; // Use cacheKey
                console.log("Non-streamed response cached (no metadata) for key:", cacheKey);
                responseOutput.innerHTML = renderFormattedText(rawText); // Render formatted
                finalizeUI(true); // Finalize UI
            }
        } catch (error) {
            console.error('Error submitting prompt:', error);
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.classList.remove('hidden');
            finalizeUI(false); // Finalize UI on fetch/network error
        }
    });

    // --- Handle Sorting ---
    sortButtons.forEach(button => {
        button.addEventListener('click', () => {
            const sortBy = button.dataset.sort;
            let order = button.dataset.order;

            // Update active button style first
            sortButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // --- Toggle Sort Order Logic ---
            // If clicking the same button, toggle the order
            if (currentSort.field === sortBy) {
                order = currentSort.order === 'asc' ? 'desc' : 'asc';
                // Update the button's data-order attribute for visual consistency (optional)
                button.dataset.order = order;
                console.log(`Toggled order to: ${order}`);
            } else {
                // If switching field, reset order based on button's default data-order
                order = button.dataset.order;
                console.log(`Switched sort field to: ${sortBy}, initial order: ${order}`);
            }
            // --------------------------- //

            // Update current sort state
            currentSort = { field: sortBy, order: order };

            // Fetch essays with new sorting
            renderEssays(allEssays);
        });
    });

    // --- Handle Checkbox Change ---
    thinkingCheckbox.addEventListener('change', () => {
        const thinkingEnabled = thinkingCheckbox.checked;
        console.log("Thinking checkbox changed to:", thinkingEnabled);
        try {
            const url = new URL(window.location);
            url.searchParams.set('thinking', thinkingEnabled.toString());
            // Use replaceState to update URL without adding a new history entry
            history.replaceState(history.state, '', url.toString());
            console.log("URL updated on checkbox change:", url.toString());
        } catch (e) {
            console.error("Error updating URL on checkbox change:", e);
        }

        // Optional: Clear cache for the current prompt as the result might change
        // const currentPrompt = promptInput.value.trim();
        // if (currentPrompt) {
        //     const cacheKeyTrue = getCacheKey(currentPrompt, true);
        //     const cacheKeyFalse = getCacheKey(currentPrompt, false);
        //     delete responseCache[cacheKeyTrue];
        //     delete responseCache[cacheKeyFalse];
        //     console.log(`Cleared cache entries for prompt '${currentPrompt}' due to thinking change.`);
        // }
    });

    // --- Initial Load Logic ---
    function initializePage() {
        console.log("Initializing page...");
        // Check for URL parameters on initial load
        const urlParams = new URLSearchParams(window.location.search);
        const promptFromUrl = urlParams.get('prompt');
        const thinkingFromUrl = urlParams.get('thinking'); // Get thinking state

        // Set checkbox state based on URL param
        if (thinkingFromUrl !== null) {
            thinkingCheckbox.checked = thinkingFromUrl === 'true';
            console.log(`Thinking checkbox set to ${thinkingCheckbox.checked} from URL.`);
        } else {
            console.log("No thinking state in URL, using default (checked).");
            // Default is checked in HTML, so no action needed unless we want to force it
            // thinkingCheckbox.checked = true;
        }

        if (promptFromUrl) {
            console.log("Found prompt in URL:", promptFromUrl);
            // Prefill the input box
            promptInput.value = decodeURIComponent(promptFromUrl); // Decode just in case
            // Update character count
            promptInput.dispatchEvent(new Event('input'));
            // Automatically trigger form submission
            // Use a small timeout to ensure the DOM is fully ready and rendering isn't blocked
            setTimeout(() => {
                console.log("Submitting prompt from URL...");
                promptForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
            }, 100); // 100ms delay, adjust if needed
        } else {
            console.log("No prompt found in URL, loading default essays.");
            // Fetch essays as normal if no prompt in URL
            fetchEssays();
        }

        // Initialize char count display regardless
        promptInput.dispatchEvent(new Event('input'));
    }

    // Run initialization logic
    initializePage();
});
