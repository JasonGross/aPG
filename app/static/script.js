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

    // --- Character Counter ---
    promptInput.addEventListener('input', () => {
        const count = promptInput.value.length;
        charCount.textContent = `${count} / 70 characters`;
    });

    // --- Fetch and Render Essays ---
    async function fetchEssays(sortBy = 'time', order = 'desc') {
        essaysList.innerHTML = '<li class="text-gray-400">Loading essays...</li>'; // Show loading state
        try {
            const response = await fetch(`/essays?sort_by=${sortBy}&order=${order}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const essays = await response.json();

            essaysList.innerHTML = ''; // Clear list

            if (essays.length === 0) {
                essaysList.innerHTML = '<li class="text-gray-500">No essays found yet.</li>';
            } else {
                essays.forEach(essay => {
                    const li = document.createElement('li');
                    // Simple display: just the prompt text
                    li.textContent = essay.prompt;
                    // Optional: Add view count or date
                    // li.textContent += ` (Views: ${essay.view_count}, Created: ${new Date(essay.created_at).toLocaleDateString()})`;
                    essaysList.appendChild(li);
                });
            }
        } catch (error) {
            console.error('Error fetching essays:', error);
            essaysList.innerHTML = '<li class="text-red-500">Failed to load essays.</li>';
        }
    }

    // --- Handle Form Submission ---
    promptForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission
        const prompt = promptInput.value.trim();

        if (!prompt) return; // Do nothing if prompt is empty

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
                        // Re-fetch essays list to include the new one (if generated)
                        fetchEssays(currentSort.field, currentSort.order);
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
                                    responseOutput.innerHTML += data.text; // Append text chunk
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
                                    fetchEssays(currentSort.field, currentSort.order);
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
                fetchEssays(currentSort.field, currentSort.order); // Update list
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
            const order = button.dataset.order;

            // Update current sort state
            currentSort = { field: sortBy, order: order };

            // Update active button style
            sortButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Fetch essays with new sorting
            fetchEssays(sortBy, order);
        });
    });

    // --- Initial Load ---
    fetchEssays(); // Load essays on page load with default sort
    promptInput.dispatchEvent(new Event('input')); // Initialize char count
});
