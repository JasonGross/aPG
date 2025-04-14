import os
import asyncio
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client

# Removed httpx import as we'll use the anthropic client directly
from dotenv import load_dotenv
import json
from pydantic import (
    BaseModel,
)  # For potential request body validation if not using Forms

# --- Import Anthropic ---
from anthropic import (
    AsyncAnthropic,
    APIError,
)  # Import Anthropic client and specific errors

# Load environment variables from .env file for local development
load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv(
    "SUPABASE_SERVICE_KEY"
)  # Use service key for backend operations
# ANTHROPIC_API_KEY is the standard env var, but we use LLM_API_KEY from previous steps
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# LLM_API_ENDPOINT is not needed when using the official client library

# --- Initialize Supabase Client ---
try:
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    else:
        print("Warning: Supabase URL or Key not found in environment variables.")
        supabase = None
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    supabase = None  # Set to None to indicate failure

# --- Initialize Anthropic Client ---
try:
    if ANTHROPIC_API_KEY:
        # Use the API key from the environment variable LLM_API_KEY
        anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    else:
        print("Warning: Anthropic API Key (LLM_API_KEY) not found.")
        anthropic_client = None
except Exception as e:
    print(f"Error initializing Anthropic client: {e}")
    anthropic_client = None

# --- Initialize FastAPI App ---
app = FastAPI()

# --- Mount Static Files (CSS, JS) ---
script_dir = os.path.dirname(__file__)
static_dir = os.path.join(script_dir, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)  # Create static dir if it doesn't exist
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- Configure Templates ---
templates_dir = os.path.join(script_dir, "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)  # Create templates dir if it doesn't exist
templates = Jinja2Templates(directory=templates_dir)


# --- Helper Functions ---
def truncate_prompt(text: str, max_length: int = 70) -> str:
    """Truncates text to a maximum length."""
    return text[:max_length]


async def get_llm_stream(prompt: str):
    """
    Gets streaming response from Anthropic Claude API.
    Yields text chunks.
    """
    if not anthropic_client:
        print("Anthropic client not initialized.")
        yield f"data: {json.dumps({'error': 'LLM service not configured.'})}\n\n"
        return

    full_llm_prompt = f"Write a Paul Graham essay about {prompt}"
    system_prompt = "You are an AI assistant that writes essays in the style of Paul Graham. Focus on insights about startups, technology, programming, and contrarian thinking. Be concise and clear."
    # Using Claude 3.5 Sonnet model ID
    model_name = "claude-3-5-sonnet-20240620"
    max_tokens_to_sample = 2048  # Adjust as needed

    print(f"Sending to Claude ({model_name}): {full_llm_prompt}")  # For debugging

    try:
        # Use the Messages streaming API
        async with anthropic_client.messages.stream(
            model=model_name,
            max_tokens=max_tokens_to_sample,
            system=system_prompt,
            messages=[{"role": "user", "content": full_llm_prompt}],
        ) as stream:
            # Iterate through the stream events asynchronously
            async for event in stream:
                # Check for text delta events
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    text_chunk = event.delta.text
                    # print(f"Received chunk: {text_chunk}") # Debugging
                    yield text_chunk  # Yield the raw text chunk

    except APIError as e:
        print(f"Anthropic API Error: {e}")
        yield f"data: {json.dumps({'error': f'LLM API Error: {e.status_code} - {e.message}'})}\n\n"
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        yield f"data: {json.dumps({'error': f'Failed to get response from LLM: {e}'})}\n\n"


async def stream_and_save_essay(truncated_prompt: str):
    """
    Streams response from LLM, yields chunks for the client (as SSE events),
    and saves the full response to Supabase upon completion.
    """
    full_response = ""
    error_occurred = False
    try:
        async for chunk in get_llm_stream(truncated_prompt):
            # Check if the chunk indicates an error (yielded by get_llm_stream)
            if isinstance(chunk, str) and chunk.startswith('data: {"error":'):
                yield chunk  # Propagate error SSE event to client
                print(f"LLM Stream Error reported: {chunk}")
                error_occurred = True
                # Don't break here, let get_llm_stream finish if it yields more details
                # Break or return after the loop if error_occurred is True
                continue  # Skip processing this chunk as text

            # Accumulate response and yield chunk to client
            full_response += chunk
            # Format as Server-Sent Event (SSE)
            yield f"data: {json.dumps({'text': chunk})}\n\n"
            await asyncio.sleep(0.01)  # Small delay to allow client processing

        # If an error was yielded during the stream, don't save or send 'end'
        if error_occurred:
            print("Skipping save due to previous stream error.")
            return

        # --- Save to Supabase after successful streaming ---
        if supabase and full_response:
            try:
                # Check again before inserting, in case of race condition
                check_resp = (
                    supabase.table("essays")
                    .select("id")
                    .eq("prompt", truncated_prompt)
                    .limit(1)
                    .execute()
                )
                if not check_resp.data:
                    insert_resp = (
                        supabase.table("essays")
                        .insert(
                            {
                                "prompt": truncated_prompt,
                                "response": full_response,
                                "view_count": 1,  # Initial view count
                            }
                        )
                        .execute()
                    )
                    print(f"Saved new essay for prompt: {truncated_prompt}")
                else:
                    print(
                        f"Essay for '{truncated_prompt}' already exists (checked before insert). Not saving again."
                    )

            except Exception as e:
                # Handle potential unique constraint violation if race condition occurs
                # The check above makes this less likely, but keep handling just in case.
                if "duplicate key value violates unique constraint" in str(e):
                    print(
                        f"Race condition? Essay for '{truncated_prompt}' already exists."
                    )
                else:
                    print(f"Error saving essay to Supabase: {e}")
                    # Optionally yield an error message back to client if critical
                    yield f"data: {json.dumps({'error': 'Failed to save essay.'})}\n\n"
                    error_occurred = True  # Mark error occurred

        # Signal stream end only if no errors occurred
        if not error_occurred:
            yield f"data: {json.dumps({'end': True})}\n\n"

    except Exception as e:
        print(f"Error during streaming/saving: {e}")
        # Send error as SSE data event
        yield f"data: {json.dumps({'error': f'An error occurred: {e}'})}\n\n"


# --- API Endpoints ---


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    if not os.path.exists(os.path.join(templates_dir, "index.html")):
        raise HTTPException(status_code=404, detail="index.html not found")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask_paul_graham(prompt: str = Form(...)):
    """
    Handles prompt submission, checks cache, calls LLM, streams response.
    Uses Server-Sent Events (SSE).
    """
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if not supabase:
        # Return SSE error if DB is down
        async def db_error_stream():
            yield f"data: {json.dumps({'error': 'Database connection not available.'})}\n\n"

        return StreamingResponse(
            db_error_stream(), media_type="text/event-stream", status_code=503
        )
    if not anthropic_client:
        # Return SSE error if LLM client isn't configured
        async def llm_error_stream():
            yield f"data: {json.dumps({'error': 'LLM service not configured.'})}\n\n"

        return StreamingResponse(
            llm_error_stream(), media_type="text/event-stream", status_code=503
        )

    truncated = truncate_prompt(prompt)

    try:
        # Check if essay already exists
        response = (
            supabase.table("essays")
            .select("response, view_count")
            .eq("prompt", truncated)
            .limit(1)
            .execute()
        )
        existing_essay = response.data

        if existing_essay:
            print(f"Cache hit for prompt: {truncated}")
            essay_data = existing_essay[0]
            saved_response = essay_data["response"]
            current_views = essay_data["view_count"]

            # Increment view count
            try:
                supabase.table("essays").update({"view_count": current_views + 1}).eq(
                    "prompt", truncated
                ).execute()
            except Exception as e:
                print(f"Error updating view count: {e}")  # Log error but continue

            # Stream the cached response word by word (or chunk by chunk) as SSE
            async def stream_cached():
                # Split into smaller chunks for smoother streaming simulation
                chunk_size = 20  # Number of characters per chunk
                for i in range(0, len(saved_response), chunk_size):
                    chunk = saved_response[i : i + chunk_size]
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0.01)  # Simulate streaming delay
                yield f"data: {json.dumps({'end': True})}\n\n"  # Signal end

            return StreamingResponse(stream_cached(), media_type="text/event-stream")

        else:
            print(f"Cache miss for prompt: {truncated}. Calling LLM.")
            # Stream from LLM and save upon completion
            return StreamingResponse(
                stream_and_save_essay(truncated), media_type="text/event-stream"
            )

    except Exception as e:
        print(f"Error in /ask endpoint: {e}")

        # Return error as SSE event
        async def error_stream():
            yield f"data: {json.dumps({'error': f'Server error: {e}'})}\n\n"

        return StreamingResponse(
            error_stream(), media_type="text/event-stream", status_code=500
        )


@app.get("/essays", response_class=JSONResponse)
async def get_essays(sort_by: str = "time", order: str = "desc"):
    """Fetches the list of saved essay prompts based on sorting preferences."""
    if not supabase:
        return JSONResponse(
            content={"error": "Database connection not available."}, status_code=503
        )

    valid_sort_by = {"time": "created_at", "views": "view_count", "alpha": "prompt"}
    valid_order = {"asc": True, "desc": False}

    sort_column = valid_sort_by.get(sort_by, "created_at")
    ascending = valid_order.get(order, False)

    try:
        response = (
            supabase.table("essays")
            .select("prompt, created_at, view_count")
            .order(sort_column, ascending=ascending)
            .execute()
        )

        # Convert Timestamps to ISO format strings for JSON serialization
        essays_data = []
        if response.data:
            for row in response.data:
                row["created_at"] = (
                    row["created_at"].isoformat() if row.get("created_at") else None
                )
                essays_data.append(row)
            return JSONResponse(content=essays_data)
        else:
            return JSONResponse(content=[])  # Return empty list if no essays

    except Exception as e:
        print(f"Error fetching essays: {e}")
        return JSONResponse(
            content={"error": f"Failed to fetch essays: {e}"}, status_code=500
        )


# --- Optional: Add simple health check ---
@app.get("/health")
async def health_check():
    db_status = "ok" if supabase else "unavailable"
    llm_status = "ok" if anthropic_client else "unavailable"
    return {"status": "ok", "database": db_status, "llm_service": llm_status}


# Note: For Hugging Face Spaces deployment using Dockerfile,
# the CMD instruction will run uvicorn. No need for __main__ block.
