// src/index.ts
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';

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

    // Add bindings for KV, R2, etc. if needed later
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
        const supabase = getSupabaseClient(env);
        const anthropic = getAnthropicClient(env);

        // --- Basic Routing ---
        if (url.pathname === '/ask' && request.method === 'POST') {
            return handleAsk(request, env, supabase, anthropic);
        }

        if (url.pathname === '/essays' && request.method === 'GET') {
            return handleEssays(request, env, supabase);
        }

        // Add /health or other routes if needed
        // if (url.pathname === '/health') { ... }

        // Default 404 for other paths
        return new Response('Not Found', { status: 404 });
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
        // --- Query Prompts and Aggregate View Counts ---
        // Note: Using an RPC function in Supabase is recommended for efficiency.
        // Replicating the multi-query approach from Python:

        // 1. Fetch all prompts
        const { data: promptsData, error: promptsError } = await supabase
            .from('prompts')
            .select('prompt_id, short_description, created_at');

        if (promptsError) throw promptsError;
        if (!promptsData || promptsData.length === 0) {
            return new Response(JSON.stringify([]), { headers: { 'Content-Type': 'application/json' } });
        }

        const promptsMap: { [id: string]: any } = {};
        const promptIds: string[] = [];
        promptsData.forEach(p => {
            promptsMap[p.prompt_id] = p;
            promptIds.push(p.prompt_id);
        });

        // 2. Fetch response IDs linked to these prompts
        const { data: responsesIdsData, error: responsesError } = await supabase
            .from('responses')
            .select('prompt_id, response_id')
            .in('prompt_id', promptIds);

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

        // 3. Fetch view counts for these response IDs
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
            return {
                prompt: promptInfo.short_description,
                created_at: promptInfo.created_at,
                view_count: totalViews,
            };
        });

        console.log(`Processed ${finalData.length} prompts.`);
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

    // Start processing async, but return the readable stream immediately
    (async () => {
        const writer = writable.getWriter();
        try {
            // --- Get or Create Model Params ---
            const { data: paramsData, error: paramsError } = await supabase
                .from('model_params')
                .upsert(
                    { model_name, system_prompt, max_tokens },
                    { onConflict: 'model_name, system_prompt, max_tokens', ignoreDuplicates: false } // Ensure conflict returns data
                )
                .select('params_id')
                .single(); // Expect single row

            if (paramsError || !paramsData) throw paramsError || new Error("Failed to upsert/find model_params.");
            const params_id = paramsData.params_id;
            console.log("Got params_id:", params_id);

            // --- Find or Create Prompt ---
            const { data: promptData, error: promptError } = await supabase
                .from('prompts')
                .upsert(
                    { short_description, prompt_text },
                    { onConflict: 'prompt_text', ignoreDuplicates: false }
                )
                .select('prompt_id, created_at')
                .single(); // Expect single row

            if (promptError || !promptData) throw promptError || new Error("Failed to upsert/find prompt.");
            const prompt_id = promptData.prompt_id;
            const prompt_created_at = promptData.created_at; // Use for logic if needed
            console.log("Got prompt_id:", prompt_id);


            // --- Check for Existing Response (Latest) ---
            const { data: latestResponseData, error: responseError } = await supabase
                .from('responses')
                .select('response_id, response_text')
                .eq('prompt_id', prompt_id)
                .order('response_created_at', { ascending: false })
                .limit(1)
                .maybeSingle(); // Can be null if no response exists

            if (responseError) throw responseError;

            if (latestResponseData) {
                console.log("Found existing response:", latestResponseData.response_id);
                // --- Stream Cached Response ---
                const response_id = latestResponseData.response_id;
                const cachedText = latestResponseData.response_text;

                // Record View
                const { error: viewError } = await supabase.from('view_counts').insert({ response_id });
                if (viewError) console.error("Failed to record view for cached response:", viewError);

                // Stream the cached text (can be chunked if large)
                await writer.write(cachedText); // Assuming createSSEEncoder handles string -> {text: ...}
                await writer.write({ end: true }); // Use object for end signal
                console.log("Finished streaming cached response.");

            } else {
                // --- Generate New Response via Anthropic ---
                console.log("No existing response found. Calling Anthropic...");
                const messages: Anthropic.Messages.MessageParam[] = [{ role: 'user', content: prompt_text }];
                let fullResponseText = "";

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
                            await writer.write(textChunk); // Stream chunk
                        }
                        // Handle other event types if needed (e.g., message_stop)
                        if (event.type === 'message_stop') {
                            console.log("Anthropic stream stopped.");
                        }
                    }

                    // --- Save the New Response ---
                    console.log("Anthropic stream finished. Saving response...");
                    const { data: insertResponseData, error: insertError } = await supabase
                        .from('responses')
                        .insert({ prompt_id, params_id, response_text: fullResponseText })
                        .select('response_id')
                        .single();

                    if (insertError || !insertResponseData) throw insertError || new Error("Failed to insert new response.");
                    const new_response_id = insertResponseData.response_id;
                    console.log("Saved new response:", new_response_id);

                    // Record Initial View
                    const { error: viewError } = await supabase.from('view_counts').insert({ response_id: new_response_id });
                    if (viewError) console.error("Failed to record initial view:", viewError);

                    await writer.write({ end: true }); // Signal end after saving
                    console.log("Finished streaming and saving new response.");

                } catch (llmError: any) {
                    console.error("Error during Anthropic call or response saving:", llmError);
                    await writer.write({ error: `LLM or save error: ${llmError.message}` });
                    // Do not close writer here, let finally block handle it
                }
            }

        } catch (error: any) {
            console.error("Error in /ask handler:", error);
            // Ensure error message is sent even if stream partially wrote data
            try {
                await writer.write({ error: `Server error: ${error.message}` });
            } catch (writeError) {
                console.error("Failed to write error to stream:", writeError);
            }
        } finally {
            // Ensure the writer is closed in all cases (success, error)
            try {
                await writer.close();
            } catch (closeError) {
                console.error("Error closing SSE writer:", closeError);
            }
        }
    })(); // IIFE to start async processing

    // Return the readable side of the stream immediately
    return new Response(encodedStream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        },
    });
}
