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

    // --- Character Counter ---
    promptInput.addEventListener('input', () => {
        const count = promptInput.value.length;
        charCount.textContent = `${count} / 70 characters`;
    });

    // --- Fetch and Render Essays ---
    async function fetchEssays(sortBy = 'time', order = 'desc') {
        console.log(`Fetching essays: sort=${sortBy}, order=${order}`); // Log fetch start
        essaysList.innerHTML = '<li class="text-gray-400">Loading essays...</li>'; // Show loading state
        try {
            const response = await fetch(`/essays?sort_by=${sortBy}&order=${order}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const essays = await response.json();
            console.log("Fetched essays data:", essays); // Log the raw fetched data

            essaysList.innerHTML = ''; // Clear list

            if (!Array.isArray(essays)) {
                console.error("Invalid data received from /essays endpoint. Expected an array.", essays);
                throw new Error("Received invalid data from server.");
            }

            if (essays.length === 0) {
                essaysList.innerHTML = '<li class="text-gray-500">No essays found yet.</li>';
            } else {
                console.log("Rendering essays..."); // Log before starting loop
                essays.forEach((essay, index) => {
                    console.log(`Rendering essay ${index + 1}:`, essay); // Log each essay object
                    const li = document.createElement('li');
                    // Use essay.prompt, provide fallback if null/undefined
                    const promptText = essay.prompt ? essay.prompt : '[No prompt text]';
                    li.textContent = promptText;
                    // Optional: Add view count or date (add checks for null values)
                    // const createdAt = essay.created_at ? new Date(essay.created_at).toLocaleDateString() : 'N/A';
                    // const viewCount = essay.view_count !== undefined ? essay.view_count : 'N/A';
                    // li.textContent += ` (Views: ${viewCount}, Created: ${createdAt})`;

                    // Make the list item clickable to load the prompt
                    li.classList.add('cursor-pointer', 'hover:text-orange-700');
                    li.addEventListener('click', () => {
                        // Set the input value and simulate submission
                        promptInput.value = essay.prompt; // Use the original prompt text
                        promptInput.dispatchEvent(new Event('input')); // Update char count
                        promptForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
                    });

                    essaysList.appendChild(li);
                    console.log(`Appended item ${index + 1} to the list.`); // Confirm appendChild worked
                });
                console.log("Finished rendering essays."); // Log after loop completes
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
