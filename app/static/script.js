// static/script.js

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

    let currentSort = { field: 'time', order: 'desc' }; // Default sort
    let eventSource = null; // To hold the EventSource connection
    let isFirstChunk = false; // Flag to track the first text chunk
    let allEssays = []; // Store all fetched essays locally

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
        // Show loading state in the list while fetching
        essaysList.innerHTML = '<li class="text-gray-400">Loading essays...</li>';
        try {
            // Fetch from the simplified endpoint (no sort params)
            const response = await fetch(`/essays`);
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
        event.preventDefault(); // Prevent default form submission
        const prompt = promptInput.value.trim();

        if (!prompt) return; // Do nothing if prompt is empty

        // --- Update URL with the submitted prompt ---
        try {
            const url = new URL(window.location);
            url.searchParams.set('prompt', prompt);
            history.pushState({ prompt: prompt }, '', url.toString());
            console.log("URL updated on submit to:", url.toString());
        } catch (e) {
            console.error("Error updating URL on submit:", e);
        }
        // ------------------------------------------ //

        // Close any existing EventSource connection
        if (eventSource) {
            eventSource.close();
        }

        // UI updates: disable input/button, show loading, clear previous results
        promptInput.disabled = true;
        submitButton.disabled = true;
        submitButton.textContent = 'Generating...';
        loadingIndicator.classList.remove('hidden');
        responseOutput.innerHTML = ''; // Clear previous output
        errorMessage.classList.add('hidden'); // Hide previous errors
        isFirstChunk = true; // Reset flag for new request

        try {
            // Use EventSource for Server-Sent Events
            eventSource = new EventSource(`/ask?prompt=${encodeURIComponent(prompt)}`, { method: 'POST' }); // NOTE: EventSource uses GET by default, FastAPI route needs POST. This is tricky.
            // Standard fetch is better suited for POST + Streaming body response. Let's refactor to use fetch.

            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded', // FastAPI Form expects this
                },
                body: `prompt=${encodeURIComponent(prompt)}`
            });

            if (!response.ok) {
                // Try to read error message from backend if available
                let errorData = { message: `HTTP error! status: ${response.status}` };
                try {
                    errorData = await response.json();
                } catch (e) { /* Ignore if response is not JSON */ }
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }

            // Check if response is event-stream (SSE)
            if (response.headers.get('content-type')?.includes('text/event-stream')) {
                // Handle SSE stream with fetch
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                reader.read().then(function processText({ done, value }) {
                    if (done) {
                        console.log("Stream complete");
                        // Re-fetch all essays to include the new one (if generated)
                        // This will store the updated list and re-render with current sort
                        fetchEssays();
                        return;
                    }

                    const chunk = decoder.decode(value, { stream: true });
                    // Process potential multiple events in a single chunk
                    const lines = chunk.split('\n');
                    lines.forEach(line => {
                        if (line.startsWith('data:')) {
                            try {
                                const data = JSON.parse(line.substring(5).trim());
                                if (data.text) {
                                    // Escape HTML to prevent injection
                                    const escapedText = data.text
                                        .replace(/&/g, '&amp;')
                                        .replace(/</g, '&lt;')
                                        .replace(/>/g, '&gt;')
                                        .replace(/"/g, '&quot;')
                                        .replace(/'/g, '&#039;');

                                    let formattedOutput = "";
                                    if (isFirstChunk) {
                                        // Find the index of the first occurrence of one or more newlines
                                        const firstParaBreakIndex = escapedText.search(/\n+/);

                                        if (firstParaBreakIndex !== -1) {
                                            // Extract title (before the first paragraph break)
                                            const title = escapedText.substring(0, firstParaBreakIndex);
                                            // Extract the rest of the text (including the newlines that caused the break)
                                            const restOfText = escapedText.substring(firstParaBreakIndex);

                                            // Format: Title span + replaced newlines in the rest
                                            formattedOutput = `<span class="essay-title">${title}</span>${restOfText.replace(/\n+/g, '<br><br>')}`;
                                            isFirstChunk = false; // Only process the first chunk this way
                                        } else {
                                            // If the first chunk has no newline sequence, treat the whole chunk as the title
                                            const title = escapedText; // No need to replace newlines here if none exist
                                            formattedOutput = `<span class="essay-title">${title}</span>`;
                                            // Subsequent chunks will add <br> if needed
                                        }
                                    } else {
                                        // For subsequent chunks, just escape and replace newlines
                                        formattedOutput = escapedText.replace(/\n+/g, '<br><br>');
                                    }
                                    responseOutput.innerHTML += formattedOutput; // Append safely formatted text chunk

                                } else if (data.error) {
                                    console.error("SSE Error:", data.error);
                                    errorMessage.textContent = `Error: ${data.error}`;
                                    errorMessage.classList.remove('hidden');
                                    // Optionally close the stream reader on error
                                    reader.cancel();
                                } else if (data.end) {
                                    console.log("SSE Stream ended by server.");
                                    // Stream ended signal received
                                    reader.cancel(); // Close the reader
                                    // Re-fetch all essays after successful generation
                                    fetchEssays();
                                }
                            } catch (e) {
                                console.error("Error parsing SSE data:", e, "Line:", line);
                            }
                        }
                    });

                    // Continue reading
                    reader.read().then(processText);
                }).catch(error => {
                    console.error("Stream reading error:", error);
                    errorMessage.textContent = `Stream reading error: ${error.message}`;
                    errorMessage.classList.remove('hidden');
                });

            } else {
                // Handle non-streaming response (shouldn't happen with current backend logic)
                const text = await response.text();
                responseOutput.textContent = text;
                fetchEssays(); // Update list
            }

        } catch (error) {
            console.error('Error submitting prompt:', error);
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.classList.remove('hidden');
        } finally {
            // Re-enable form elements regardless of success/failure AFTER stream ends
            // Need to move this logic to inside the stream handling completion/error
            promptInput.disabled = false;
            submitButton.disabled = false;
            submitButton.textContent = 'Generate Essay';
            loadingIndicator.classList.add('hidden');
            // Note: Re-enabling is handled more accurately within the stream processing logic (on done/error/end)
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

    // --- Initial Load Logic ---
    function initializePage() {
        console.log("Initializing page...");
        // Check for URL parameters on initial load
        const urlParams = new URLSearchParams(window.location.search);
        const promptFromUrl = urlParams.get('prompt');

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
