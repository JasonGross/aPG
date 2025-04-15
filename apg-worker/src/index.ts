// src/index.ts
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';
import type { ExecutionContext } from '@cloudflare/workers-types';

// Define the structure of environment variables/secrets
export interface Env {
    SUPABASE_URL: string;
    SUPABASE_SERVICE_KEY: string;
    ANTHROPIC_API_KEY: string;

    // Non-secret vars
    LLM_MODEL_NAME: string;
    LLM_SYSTEM_PROMPT: string;
    LLM_MAX_TOKENS: string;
    LLM_PROMPT_TEMPLATE: string;
    RESTRICT_TO_CURRENT_MODEL: boolean;
    RESTRICT_TO_CURRENT_PROMPT: boolean;

    // Add bindings for KV, R2, etc. if needed later
}

// --- CORS Configuration ---
const ALLOWED_ORIGINS: string[] = [
    'https://apg-6do.pages.dev',
    'https://askpg.ai',
    'https://askpaulgraham.ai',
    // Add localhost for local development if needed:
    'http://localhost:8000', // Example port
    'http://127.0.0.1:8000'
];

const BASE_CORS_HEADERS = {
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400', // Optional: Cache preflight response for 1 day
};

// --- Helper: Add CORS headers to a Response ---
function addCorsHeaders(response: Response, origin: string): Response {
    response = new Response(response.body, response);
    // Set base headers
    Object.entries(BASE_CORS_HEADERS).forEach(([key, value]) => {
        response.headers.set(key, value);
    });
    // Set the specific allowed origin
    response.headers.set('Access-Control-Allow-Origin', origin);
    return response;
}

// --- Helper: Handle OPTIONS Preflight Requests ---
function handleOptions(request: Request): Response {
    const origin = request.headers.get('Origin');

    // Check if origin is allowed
    if (origin && ALLOWED_ORIGINS.includes(origin)) {
        // Check if it's a valid preflight request
        if (
            request.headers.get('Access-Control-Request-Method') !== null &&
            request.headers.get('Access-Control-Request-Headers') !== null
        ) {
            // Handle CORS preflight requests with dynamic origin.
            const headers = {
                ...BASE_CORS_HEADERS,
                'Access-Control-Allow-Origin': origin, // Echo the specific origin
            };
            return new Response(null, {
                status: 204, // No Content
                headers: headers,
            });
        }
    }

    // If origin not allowed or not a valid preflight, handle as a standard OPTIONS request
    // Or return 403, but a simple response is often sufficient.
    return new Response(null, {
        status: 204, // Typically OPTIONS requests don't need a body
        headers: {
            Allow: 'GET, POST, OPTIONS', // Methods allowed on the worker
        },
    });
}

// --- Helper: Initialize Clients (Consider Singleton if needed) ---
function getSupabaseClient(env: Env): SupabaseClient | null {
    if (!env.SUPABASE_URL || !env.SUPABASE_SERVICE_KEY) {
        console.error("Supabase URL or Key missing in environment.");
        return null;
    }
    try {
        // Note: Cloudflare integration might involve specific options
        // depending on how you set it up (e.g., using fetch from executionContext).
        // Check Supabase Cloudflare integration docs for specifics.
        // This is a basic initialization.
        return createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY, {
            // Example: You might need to pass the fetch implementation from the environment
            // global: { fetch: fetch }
            // auth: { persistSession: false } // Recommended for stateless workers
        });
    } catch (error) {
        console.error("Error initializing Supabase client:", error);
        return null;
    }
}

function getAnthropicClient(env: Env): Anthropic | null {
    if (!env.ANTHROPIC_API_KEY) {
        console.error("Anthropic API Key missing in environment.");
        return null;
    }
    try {
        return new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });
    } catch (error) {
        console.error("Error initializing Anthropic client:", error);
        return null;
    }
}


// --- Helper: SSE Stream Formatting ---
function createSSEEncoder() {
    return new TransformStream<string | object, Uint8Array>({
        transform(chunk, controller) {
            let message = '';
            if (typeof chunk === 'object') {
                message = `data: ${JSON.stringify(chunk)}\n\n`;
            } else {
                // Assume string chunk needs formatting for SSE 'text' field
                message = `data: ${JSON.stringify({ text: chunk })}\n\n`;
            }
            controller.enqueue(new TextEncoder().encode(message));
        },
    });
}

function createErrorSSEEvent(message: string): string {
    return `data: ${JSON.stringify({ error: message })}\n\n`;
}
function createEndSSEEvent(): string {
    return `data: ${JSON.stringify({ end: true })}\n\n`;
}

// --- Main Fetch Handler ---
export default {
    async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
        const url = new URL(request.url);
        const origin = request.headers.get("Origin");

        // --- Handle OPTIONS Preflight Requests ---
        if (request.method === 'OPTIONS') {
            return handleOptions(request);
        }

        // --- Check Origin for non-OPTIONS requests ---
        if (!origin || !ALLOWED_ORIGINS.includes(origin)) {
            console.warn(`Origin ${origin} not allowed.`);
            // Return a forbidden response if origin is missing or not allowed
            return new Response(JSON.stringify({ error: 'Forbidden - Origin not allowed' }), {
                status: 403,
                headers: { 'Content-Type': 'application/json' }
                // No CORS headers needed for a 403 response usually
            });
        }

        // --- Initialize Clients ---
        const supabase = getSupabaseClient(env);
        const anthropic = getAnthropicClient(env);
        let response: Response; // Declare response variable

        // --- Basic Routing ---
        try {
            if (url.pathname === '/ask' && request.method === 'POST') {
                response = await handleAsk(request, env, supabase, anthropic);
            } else if (url.pathname === '/essays' && request.method === 'GET') {
                response = await handleEssays(request, env, supabase);
            } else {
                response = new Response('Not Found', { status: 404 });
            }
        } catch (e: any) {
            console.error("Unhandled error in fetch handler:", e);
            // Ensure a generic error response
            response = new Response(JSON.stringify({ error: "Internal Server Error" }), {
                status: 500,
                headers: { 'Content-Type': 'application/json' }
            });
        }

        // --- Add CORS Headers Dynamically to ALL Allowed Responses ---
        // The origin check above ensures we only reach here if the origin is allowed
        return addCorsHeaders(response, origin);
    },
};

// --- /essays Handler ---
async function handleEssays(request: Request, env: Env, supabase: SupabaseClient | null): Promise<Response> {
    console.log("Handling /essays request");
    if (!supabase) {
        return new Response(JSON.stringify({ error: 'Database connection not available.' }), {
            status: 503, headers: { 'Content-Type': 'application/json' }
        });
    }

    try {
        // --- Check Model Restriction ---
        const restrictByModel = env.RESTRICT_TO_CURRENT_MODEL?.toString().toLowerCase() === 'true';
        let current_params_id: string | null = null;
        if (restrictByModel) {
            console.log("Restriction to current model enabled. Fetching current params_id...");
            const model_name = env.LLM_MODEL_NAME || 'claude-3-haiku-20240307';
            const system_prompt = env.LLM_SYSTEM_PROMPT || '';
            const max_tokens = parseInt(env.LLM_MAX_TOKENS || '1000', 10);
            const { data: paramsData, error: paramsError } = await supabase.from('model_params').select('params_id').match({ model_name, system_prompt, max_tokens }).maybeSingle();
            if (paramsError) {
                console.error("Error fetching current model_params for restriction:", paramsError);
                // Proceed without restriction if params lookup fails?
                // Or return error? Let's proceed without restriction for now.
            } else if (paramsData) {
                current_params_id = paramsData.params_id;
                console.log(`Current model params_id found: ${current_params_id}`);
            } else {
                console.warn("Current model parameters not found in DB. Cannot restrict results.");
            }
        }

        // --- Check Prompt Template Restriction ---
        const restrictByPrompt = env.RESTRICT_TO_CURRENT_PROMPT?.toString().toLowerCase() === 'true';
        let promptPrefixFilter: string | null = null;
        if (restrictByPrompt) {
            console.log("Restriction to current prompt template enabled.");
            const current_template = env.LLM_PROMPT_TEMPLATE || 'Write an essay about {short_description}';
            const placeholderIndex = current_template.indexOf('{short_description}');
            if (placeholderIndex > 0) {
                promptPrefixFilter = current_template.substring(0, placeholderIndex);
                console.log(`Filtering prompts where prompt_text starts with: "${promptPrefixFilter}"`);
            } else {
                // If the template doesn't contain the placeholder as expected, we can't reliably filter.
                console.warn(`Prompt template "${current_template}" does not contain "{short_description}" in the expected format. Cannot filter by prompt template.`);
            }
        }
        // ------------------------------------------ //

        // --- Query Prompts and Aggregate View Counts ---
        // 1. Fetch prompts (apply prompt filter if needed)
        let promptsQuery = supabase
            .from('prompts')
            .select('prompt_id, short_description, created_at');

        if (restrictByPrompt && promptPrefixFilter) {
            promptsQuery = promptsQuery.like('prompt_text', `${promptPrefixFilter}%`);
        }

        const { data: promptsData, error: promptsError } = await promptsQuery;

        if (promptsError) throw promptsError;
        if (!promptsData || promptsData.length === 0) {
            console.log("No prompts found matching the criteria.");
            return new Response(JSON.stringify([]), { headers: { 'Content-Type': 'application/json' } });
        }

        const promptsMap: { [id: string]: any } = {};
        const promptIds: string[] = [];
        promptsData.forEach(p => {
            promptsMap[p.prompt_id] = p;
            promptIds.push(p.prompt_id);
        });

        // 2. Fetch response IDs linked to these prompts (Apply model filter if needed)
        let responseQuery = supabase
            .from('responses')
            .select('prompt_id, response_id')
            .in('prompt_id', promptIds);

        if (restrictByModel && current_params_id) {
            console.log(`Applying model restriction: Only responses with params_id=${current_params_id}`);
            responseQuery = responseQuery.eq('params_id', current_params_id);
        }

        const { data: responsesIdsData, error: responsesError } = await responseQuery;

        if (responsesError) throw responsesError;

        const responseIds: string[] = [];
        const promptIdToResponseIds: { [pid: string]: string[] } = {};
        if (responsesIdsData) {
            responsesIdsData.forEach(r => {
                if (!promptIdToResponseIds[r.prompt_id]) {
                    promptIdToResponseIds[r.prompt_id] = [];
                }
                promptIdToResponseIds[r.prompt_id].push(r.response_id);
                responseIds.push(r.response_id);
            });
        }

        // 3. Fetch view counts for these (potentially filtered) response IDs
        const viewsPerResponse: { [rid: string]: number } = {};
        if (responseIds.length > 0) {
            const { data: viewsData, error: viewsError } = await supabase
                .from('view_counts')
                .select('response_id', { count: 'exact' }) // Request count
                .in('response_id', responseIds)
                .limit(1000); // Adjust limit as needed or handle pagination

            // We need to manually count if using select like this, or use RPC
            // Let's simulate manual counting from a full select (less efficient)
            const { data: allViews, error: allViewsError } = await supabase
                .from('view_counts')
                .select('response_id')
                .in('response_id', responseIds);

            if (allViewsError) throw allViewsError;

            if (allViews) {
                allViews.forEach(v => {
                    viewsPerResponse[v.response_id] = (viewsPerResponse[v.response_id] || 0) + 1;
                })
            }
        }

        // 4. Combine data
        const finalData = Object.values(promptsMap).map(promptInfo => {
            const pid = promptInfo.prompt_id;
            let totalViews = 0;
            if (promptIdToResponseIds[pid]) {
                promptIdToResponseIds[pid].forEach(rid => {
                    totalViews += (viewsPerResponse[rid] || 0);
                });
            }
            if (!restrictByModel || promptIdToResponseIds[pid]) {
                return {
                    prompt: promptInfo.short_description,
                    created_at: promptInfo.created_at,
                    view_count: totalViews,
                };
            }
            return null;
        }).filter(item => item !== null);

        console.log(`Processed ${finalData.length} prompts after potential filtering.`);
        // Return unsorted data (client-side handles sorting)
        return new Response(JSON.stringify(finalData), { headers: { 'Content-Type': 'application/json' } });

    } catch (error: any) {
        console.error("Error fetching /essays:", error);
        return new Response(JSON.stringify({ error: 'Failed to fetch essays.', details: error.message }), {
            status: 500, headers: { 'Content-Type': 'application/json' }
        });
    }
}

// --- /ask Handler ---
async function handleAsk(request: Request, env: Env, supabase: SupabaseClient | null, anthropic: Anthropic | null): Promise<Response> {
    console.log("Handling /ask request");

    // --- Service Checks ---
    if (!supabase) return new Response(createErrorSSEEvent('Database connection not available.'), { status: 503, headers: { 'Content-Type': 'text/event-stream' } });
    if (!anthropic) return new Response(createErrorSSEEvent('LLM service not configured.'), { status: 503, headers: { 'Content-Type': 'text/event-stream' } });

    // --- Get Prompt from Form Data ---
    let short_description = '';
    try {
        const formData = await request.formData();
        const promptValue = formData.get('prompt');
        if (typeof promptValue === 'string') {
            short_description = promptValue.trim();
        }
        if (!short_description) {
            return new Response(createErrorSSEEvent('Prompt cannot be empty.'), { status: 400, headers: { 'Content-Type': 'text/event-stream' } });
        }
        console.log("Received prompt (short_description):", short_description);
    } catch (e) {
        return new Response(createErrorSSEEvent('Failed to parse form data.'), { status: 400, headers: { 'Content-Type': 'text/event-stream' } });
    }


    // --- Get LLM Config from Env ---
    const model_name = env.LLM_MODEL_NAME || 'claude-3-haiku-20240307'; // Default
    const system_prompt = env.LLM_SYSTEM_PROMPT || '';
    const max_tokens = parseInt(env.LLM_MAX_TOKENS || '1000', 10);
    const prompt_template = env.LLM_PROMPT_TEMPLATE || 'Write an essay about {short_description}';
    let prompt_text = '';
    try {
        prompt_text = prompt_template.replace('{short_description}', short_description);
    } catch (e) {
        console.error("Error formatting prompt template:", e);
        prompt_text = `Write an essay about ${short_description}`; // Fallback
    }
    console.log("Using LLM Params:", { model_name, max_tokens });


    // --- SSE Stream Setup ---
    const { readable, writable } = new TransformStream();
    const encoder = createSSEEncoder();
    const encodedStream = readable.pipeThrough(encoder);

    // Start processing async
    (async () => {
        const writer = writable.getWriter();
        let streamErrorOccurred = false;
        try {
            // --- DB Operations: Get/Create Params & Prompt (Keep existing select/insert logic) ---
            let params_id: string;
            console.log("Checking for existing model_params...");
            const { data: existingParams, error: paramsSelectError } = await supabase
                .from('model_params')
                .select('params_id')
                .match({ model_name, system_prompt, max_tokens })
                .maybeSingle();

            if (paramsSelectError) {
                console.error("Error selecting model_params:", paramsSelectError);
                throw new Error(`DB error checking model params: ${paramsSelectError.message}`);
            }

            if (existingParams) {
                params_id = existingParams.params_id;
                console.log("Found existing model_params_id:", params_id);
            } else {
                console.log("No existing model_params found, creating new one...");
                const { data: newParamsData, error: paramsInsertError } = await supabase
                    .from('model_params')
                    .insert({ model_name, system_prompt, max_tokens })
                    .select('params_id')
                    .single();

                if (paramsInsertError || !newParamsData) {
                    console.error("Error inserting new model_params:", paramsInsertError);
                    throw new Error(`DB error creating model parameters: ${paramsInsertError?.message || 'Insert failed'}`);
                }
                params_id = newParamsData.params_id;
                console.log("Created new model_params_id:", params_id);
            }
            console.log("Got params_id:", params_id);

            let prompt_id: string;
            console.log("Checking for existing prompt...");
            const { data: existingPrompt, error: promptSelectError } = await supabase
                .from('prompts')
                .select('prompt_id, created_at')
                .eq('prompt_text', prompt_text)
                .maybeSingle();

            if (promptSelectError) {
                console.error("Error selecting prompt:", promptSelectError);
                throw new Error(`DB error checking prompt: ${promptSelectError.message}`);
            }

            if (existingPrompt) {
                prompt_id = existingPrompt.prompt_id;
                console.log("Found existing prompt_id:", prompt_id);
            } else {
                console.log("No existing prompt found, creating new one...");
                const { data: newPromptData, error: promptInsertError } = await supabase
                    .from('prompts')
                    .insert({ short_description, prompt_text })
                    .select('prompt_id, created_at')
                    .single();

                if (promptInsertError || !newPromptData) {
                    console.error("Error inserting new prompt:", promptInsertError);
                    throw new Error(`DB error creating prompt: ${promptInsertError?.message || 'Insert failed'}`);
                }
                prompt_id = newPromptData.prompt_id;
                console.log("Created new prompt_id:", prompt_id);
            }
            console.log("Got prompt_id:", prompt_id);
            // -------------------------------------------------------------------------------------- //

            // --- Check for Existing Response (Latest) ---
            const { data: latestResponseData, error: responseError } = await supabase
                .from('responses')
                .select('response_id, response_text')
                .eq('prompt_id', prompt_id)
                .order('response_created_at', { ascending: false })
                .limit(1)
                .maybeSingle();

            if (responseError) throw responseError;

            if (latestResponseData) {
                // --- Stream Cached Response Line by Line ---
                console.log("Found existing response:", latestResponseData.response_id);
                const response_id = latestResponseData.response_id;
                const cachedText = latestResponseData.response_text || "";

                // Record View (Best effort)
                supabase.from('view_counts').insert({ response_id }).then(({ error: viewError }) => {
                    if (viewError) console.error("Failed to record view for cached response:", viewError);
                });

                // Parameters
                const delayMs = 20; // Adjust delay between lines

                console.log(`Streaming cached response (ID: ${response_id}) line by line...`);

                const lines = cachedText.split('\n');

                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    // Send the line plus the newline character that split removed
                    // The encoder wraps this in data: {text: ...}
                    await writer.write(line + (i < lines.length - 1 ? '\n' : ''));

                    // Add a small delay
                    if (delayMs > 0) {
                        await new Promise(resolve => setTimeout(resolve, delayMs));
                    }
                }
                // -----------------------------------------

                await writer.write({ end: true }); // Signal end after all lines are sent
                console.log("Finished streaming cached response.");

            } else {
                // --- Generate New Response via Anthropic ---
                console.log("No existing response found. Calling Anthropic...");
                const messages: Anthropic.Messages.MessageParam[] = [{ role: 'user', content: prompt_text }];
                let fullResponseText = "";
                let llmStreamSuccessful = false;

                try {
                    const stream = anthropic.messages.stream({
                        model: model_name,
                        max_tokens: max_tokens,
                        system: system_prompt,
                        messages: messages,
                    });

                    for await (const event of stream) {
                        if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
                            const textChunk = event.delta.text;
                            fullResponseText += textChunk;
                            await writer.write(textChunk);
                        }
                        if (event.type === 'message_stop') {
                            console.log("Anthropic stream stopped.");
                            llmStreamSuccessful = true;
                        }
                    }
                } catch (llmError: any) {
                    console.error("Error during Anthropic stream:", llmError);
                    streamErrorOccurred = true;
                    await writer.write({ error: `LLM stream error: ${llmError.message}` });
                    llmStreamSuccessful = false;
                }

                // --- Save the New Response ONLY if LLM stream was successful ---
                if (llmStreamSuccessful) {
                    try {
                        console.log("Anthropic stream finished successfully. Saving response...");
                        const { data: insertResponseData, error: insertError } = await supabase
                            .from('responses')
                            .insert({ prompt_id, params_id, response_text: fullResponseText })
                            .select('response_id')
                            .single();

                        if (insertError || !insertResponseData) {
                            throw insertError || new Error("Failed to insert new response or get ID.");
                        }
                        const new_response_id = insertResponseData.response_id;
                        console.log("Saved new response:", new_response_id);

                        // Record Initial View (Best effort)
                        supabase.from('view_counts').insert({ response_id: new_response_id }).then(({ error: viewError }) => {
                            if (viewError) console.error("Failed to record initial view:", viewError);
                        });

                        // Signal end ONLY after successful save
                        await writer.write({ end: true });
                        console.log("Finished streaming and saving new response.");

                    } catch (saveError: any) {
                        console.error("Error saving response to Supabase:", saveError);
                        streamErrorOccurred = true;
                        await writer.write({ error: `Failed to save response: ${saveError.message}` });
                        // Do not write {end: true} if save failed
                    }
                } else {
                    // If LLM stream failed, we already sent the error, do nothing more here
                    console.log("Skipping save because LLM stream failed.");
                }
            } // end else (generate new response)

        } catch (handlerError: any) { // Catch errors from initial DB checks or response finding
            console.error("Error in /ask handler setup or response finding:", handlerError);
            streamErrorOccurred = true;
            try {
                // Try to write error even if outer try failed early
                await writer.write({ error: `Server setup error: ${handlerError.message}` });
            } catch (writeError) {
                console.error("Failed to write setup error to stream:", writeError);
            }
        } finally {
            console.log(`Closing SSE writer. Stream error occurred: ${streamErrorOccurred}`);
            // Close the writer
            try {
                await writer.close();
            } catch (closeError) {
                console.error("Error closing SSE writer:", closeError);
            }
        }
    })(); // IIFE to start async processing

    // --- Return Response Stream ---
    return new Response(encodedStream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        },
    });
}
