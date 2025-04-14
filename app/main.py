import os
import asyncio
import logging
from typing import Optional  # Import Optional for type hinting
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from supabase import Client, create_client  # type: ignore
from postgrest.types import CountMethod
from dotenv import load_dotenv
import json
from pydantic import BaseModel

# --- Import Anthropic ---
from anthropic import AsyncAnthropic, APIError

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Initialize Supabase Client ---
supabase: Optional[Client] = None  # Use Optional for type hint
try:
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully.")
    else:
        logger.warning(
            "Supabase URL or Key not found. Database functionality disabled."
        )
        # supabase is already None
except Exception as e:
    logger.exception("Error initializing Supabase client", exc_info=e)
    supabase = None  # Ensure supabase is None on exception

# --- Initialize Anthropic Client ---
anthropic_client: Optional[AsyncAnthropic] = None  # Use Optional for type hint
try:
    if ANTHROPIC_API_KEY:
        anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized successfully.")
    else:
        logger.warning(
            "Anthropic API Key (ANTHROPIC_API_KEY) not found. LLM functionality disabled."
        )
        # anthropic_client is already None
except Exception as e:
    logger.exception("Error initializing Anthropic client", exc_info=e)
    anthropic_client = None  # Ensure anthropic_client is None on exception

# --- Initialize FastAPI App ---
app = FastAPI()

# --- Mount Static Files & Templates ---
script_dir = os.path.dirname(__file__)
static_dir = os.path.join(script_dir, "static")
templates_dir = os.path.join(script_dir, "templates")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


# --- Helper Functions ---
def truncate_prompt(text: str, max_length: int = 70) -> str:
    """Truncates text to a maximum length."""
    return text[:max_length]


async def get_llm_stream(
    model_name: str, system_prompt: str, messages: list, max_tokens: int
):
    """
    Gets streaming response from Anthropic Claude API using provided arguments.
    Yields text chunks or SSE error events.
    """
    if not anthropic_client:
        logger.error("Anthropic client not initialized. Cannot call LLM.")
        yield f"data: {json.dumps({'error': 'LLM service not configured.'})}\n\n"
        return

    logger.info(f"Sending request to Claude model: '{model_name}'")
    # Log arguments (be careful with potentially large message content in production)
    # logger.debug(f"LLM Args: model={model_name}, max_tokens={max_tokens}, system='{system_prompt[:50]}...', messages={messages}")

    try:
        async with anthropic_client.messages.stream(
            model=model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        ) as stream:
            async for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    yield event.delta.text
        logger.info(f"Successfully completed LLM stream from model: '{model_name}'")
    except APIError as e:
        # Note: Accessing e.status_code might still be problematic depending on the
        # exact version of the anthropic library and the error type.
        # We'll log the whole error for now. A more robust solution might involve
        # checking the error type or attributes more carefully.
        logger.error(
            f"Anthropic API Error model='{model_name}' message='{e.message}' Error Details: {e}",
            exc_info=False,
        )
        yield f"data: {json.dumps({'error': f'LLM API Error: {e.message}'})}\n\n"  # Avoid potentially missing attributes
    except Exception as e:
        logger.exception(
            f"Error calling Anthropic API model: '{model_name}'", exc_info=e
        )
        yield f"data: {json.dumps({'error': f'Failed to get response from LLM: {e}'})}\n\n"


async def stream_and_save_new_response(
    prompt_id: str,
    model_name: str,
    system_prompt: str,
    messages: list,
    max_tokens: int,
):
    """
    Calls LLM stream with provided parameters, yields chunks for the client,
    and saves the full response along with model info to the `responses` table.
    """
    full_response = ""
    error_occurred = False

    # --- Prepare arguments dictionary for saving ---
    # Construct this based on the arguments *received* by the function
    model_arguments_to_save = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
        # Add other relevant parameters if they were passed (e.g., temperature)
    }
    # --------------------------------------------

    try:
        # Pass received arguments directly to the LLM stream function
        async for chunk in get_llm_stream(
            model_name, system_prompt, messages, max_tokens
        ):
            if isinstance(chunk, str) and chunk.startswith('data: {"error":'):
                yield chunk  # Propagate error SSE event
                logger.warning(
                    f"LLM Stream Error reported for prompt_id '{prompt_id}': {chunk}"
                )
                error_occurred = True
                continue

            full_response += chunk
            yield f"data: {json.dumps({'text': chunk})}\n\n"
            await asyncio.sleep(0.01)

        if error_occurred:
            logger.warning(
                f"Skipping save for prompt_id '{prompt_id}' due to previous stream error."
            )
            return

        # --- Save the new response to the `responses` table --- #
        # Note: model_name and model_arguments are now saved in the prompts table
        if supabase and full_response:
            logger.info(f"Attempting to save new response for prompt_id: '{prompt_id}'")
            try:
                insert_resp = (
                    supabase.table("responses")
                    .insert(
                        {
                            "prompt_id": prompt_id,
                            "response_text": full_response,
                            # "model_name": model_name,  # Removed: Belongs in prompts table
                            # "model_arguments": arguments_json, # Removed: Belongs in prompts table
                        }
                    )
                    .execute()
                )
                logger.info(
                    f"Successfully saved new response for prompt_id: '{prompt_id}'"
                )

            except Exception as e:
                logger.exception(
                    f"Error saving new response to Supabase for prompt_id '{prompt_id}'",
                    exc_info=e,
                )
                yield f"data: {json.dumps({'error': 'Failed to save new response.'})}\n\n"
                error_occurred = True

        if not error_occurred:
            logger.info(
                f"Successfully streamed and saved new response for prompt_id: '{prompt_id}'"
            )
            yield f"data: {json.dumps({'end': True})}\n\n"

    except Exception as e:
        logger.exception(
            f"Error during streaming/saving new response for prompt_id '{prompt_id}'",
            exc_info=e,
        )
        yield f"data: {json.dumps({'error': f'An error occurred during streaming/saving: {e}'})}\n\n"


# --- API Endpoints ---


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    template_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(template_path):
        logger.error(f"Template file not found at {template_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    client_host = request.client.host if request.client else "unknown"  # Safe access
    logger.info(f"Serving root page request from {client_host}")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask_paul_graham(request: Request, prompt: str = Form(...)):
    """
    Handles prompt submission. Stores original prompt as short_description.
    Uses truncated prompt (prompt_text) for uniqueness check.
    Fetches latest response if prompt_text exists, otherwise generates new response
    and stores it along with model invocation details.
    """
    client_host = request.client.host if request.client else "unknown"
    short_description = prompt.strip()
    logger.info(
        f"Received /ask request with short_description: '{short_description}' from {client_host}"
    )

    if not short_description:
        logger.warning("Received empty prompt in /ask request.")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    # --- DB/LLM Client Checks ---
    if not supabase:
        logger.error("Supabase client not available for /ask request.")

        async def db_error_stream():
            yield f"data: {json.dumps({'error': 'Database connection not available.'})}\n\n"

        return StreamingResponse(
            db_error_stream(), media_type="text/event-stream", status_code=503
        )
    if not anthropic_client:
        logger.error("Anthropic client not available for /ask request.")

        async def llm_error_stream():
            yield f"data: {json.dumps({'error': 'LLM service not configured.'})}\n\n"

        return StreamingResponse(
            llm_error_stream(), media_type="text/event-stream", status_code=503
        )
    # ---------------------------

    truncated_prompt = truncate_prompt(short_description)
    logger.info(f"Using truncated prompt_text for lookup: '{truncated_prompt}'")

    try:
        # Check if prompt_text exists in `prompts` table
        prompt_resp = (
            supabase.table("prompts")
            .select("prompt_id, view_count")
            .eq("prompt_text", truncated_prompt)
            .limit(1)
            .execute()
        )
        existing_prompt = prompt_resp.data

        if existing_prompt:
            # --- Prompt Exists ---
            prompt_data = existing_prompt[0]
            prompt_id = prompt_data["prompt_id"]
            current_views = prompt_data["view_count"]
            logger.info(
                f"Prompt exists (ID: {prompt_id}). Incrementing view count and fetching latest response."
            )

            # Increment view count
            try:
                supabase.table("prompts").update({"view_count": current_views + 1}).eq(
                    "prompt_id", prompt_id
                ).execute()
                logger.info(
                    f"Incremented view count for prompt_id '{prompt_id}' to {current_views + 1}"
                )
            except Exception as e:
                logger.error(
                    f"Error updating view count for prompt_id '{prompt_id}'", exc_info=e
                )

            # Fetch the latest response text
            latest_response_resp = (
                supabase.table("responses")
                .select("response_text")
                .eq("prompt_id", prompt_id)
                .order("response_created_at", desc=True)
                .limit(1)
                .execute()
            )

            if latest_response_resp.data:
                latest_response_text = latest_response_resp.data[0]["response_text"]
                logger.info(
                    f"Found latest response for prompt_id '{prompt_id}'. Streaming it back."
                )

                # Stream the cached/latest response
                async def stream_latest_cached():
                    chunk_size = 20
                    for i in range(0, len(latest_response_text), chunk_size):
                        chunk = latest_response_text[i : i + chunk_size]
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                        await asyncio.sleep(0.01)
                    yield f"data: {json.dumps({'end': True})}\n\n"

                return StreamingResponse(
                    stream_latest_cached(), media_type="text/event-stream"
                )
            else:
                logger.error(f"Prompt '{prompt_id}' exists, but no responses found!")

                # Option: Generate a new response for this existing prompt?
                # For now, return error. Could call stream_and_save_new_response(prompt_id, truncated_prompt) here instead.
                async def no_resp_stream():
                    yield f"data: {json.dumps({'error': 'Found prompt but no responses available.'})}\n\n"

                return StreamingResponse(
                    no_resp_stream(), media_type="text/event-stream", status_code=404
                )

        else:
            # --- Prompt Does Not Exist ---
            logger.info(
                f"Prompt_text '{truncated_prompt}' does not exist. Creating new prompt entry."
            )
            try:
                # Insert new prompt
                model_name = "claude-3-5-sonnet-20240620"
                system_prompt = "You are an AI assistant that writes essays in the style of Paul Graham. Focus on insights about startups, technology, programming, and contrarian thinking. Be concise and clear."
                max_tokens = 2048
                messages = [
                    {
                        "role": "user",
                        "content": f"Write a Paul Graham essay about {short_description}",  # Use full description for the LLM
                    }
                ]
                # ----------------------------- #

                # Insert new prompt with model details
                insert_prompt_resp = (
                    supabase.table("prompts")
                    .insert(
                        {
                            "prompt_text": truncated_prompt,
                            "short_description": short_description,
                            "view_count": 1,
                            "model_name": model_name,  # Add model name here
                            "model_arguments": messages[0][
                                "content"
                            ],  # Add arguments here (Adjust based on desired format)
                        }
                    )
                    .execute()
                )

                if insert_prompt_resp.data:
                    new_prompt_id = insert_prompt_resp.data[0]["prompt_id"]
                    logger.info(
                        f"Successfully inserted new prompt with ID: {new_prompt_id}"
                    )

                    # Generate, stream, and save the first response (including model info)
                    return StreamingResponse(
                        stream_and_save_new_response(
                            new_prompt_id,
                            model_name,
                            system_prompt,
                            messages,
                            max_tokens,
                        ),
                        media_type="text/event-stream",
                    )
                else:
                    logger.error(
                        f"Failed to insert new prompt '{truncated_prompt}'. Response: {insert_prompt_resp}"
                    )
                    raise Exception("Failed to create new prompt entry.")

            except Exception as e:
                # Handle potential race condition on prompt_text unique constraint
                if "duplicate key value violates unique constraint" in str(
                    e
                ) and "prompts_prompt_text_key" in str(e):
                    logger.warning(
                        f"Race condition? Prompt_text '{truncated_prompt}' inserted between check/insert. Recovering."
                    )
                    recover_resp = (
                        supabase.table("prompts")
                        .select("prompt_id")
                        .eq("prompt_text", truncated_prompt)
                        .limit(1)
                        .execute()
                    )
                    if recover_resp.data:
                        recovered_prompt_id = recover_resp.data[0]["prompt_id"]
                        logger.info(
                            f"Recovered prompt_id: {recovered_prompt_id}. Generating new response for existing prompt."
                        )

                        # --- Define LLM Parameters (Race Condition Recovery) --- #
                        model_name = "claude-3-5-sonnet-20240620"
                        system_prompt = "You are an AI assistant that writes essays in the style of Paul Graham. Focus on insights about startups, technology, programming, and contrarian thinking. Be concise and clear."
                        max_tokens = 2048
                        messages = [
                            {
                                "role": "user",
                                "content": f"Write a Paul Graham essay about {short_description}",  # Use full description
                            }
                        ]
                        # ----------------------------------------------------- #

                        # Generate a new response and save it, linked to the recovered prompt_id
                        # NOTE: We don't update the prompt record here as it already exists.
                        # The model details used for *this specific response* generation are saved
                        # in the responses table by stream_and_save_new_response.
                        return StreamingResponse(
                            stream_and_save_new_response(
                                recovered_prompt_id,
                                model_name,
                                system_prompt,
                                messages,
                                max_tokens,
                            ),
                            media_type="text/event-stream",
                        )
                    else:
                        logger.error(
                            f"Race condition recovery failed for prompt_text '{truncated_prompt}'."
                        )
                        raise Exception("Failed to create or recover prompt entry.")
                else:
                    logger.exception(
                        f"Error inserting new prompt with text '{truncated_prompt}'",
                        exc_info=e,
                    )
                    raise e  # Re-raise other exceptions

    except Exception as e:
        logger.exception(
            f"Error processing /ask request for prompt_text '{truncated_prompt}'",
            exc_info=e,
        )

        async def error_stream():
            yield f"data: {json.dumps({'error': f'Server error processing request.'})}\n\n"

        return StreamingResponse(
            error_stream(), media_type="text/event-stream", status_code=500
        )


@app.get("/essays", response_class=JSONResponse)
async def get_essays(sort_by: str = "time", order: str = "desc"):
    """Fetches the list of saved prompts, returning short_description."""
    logger.info(f"Received /essays request. Sort by: {sort_by}, Order: {order}")
    if not supabase:
        logger.error("Supabase client not available for /essays request.")
        return JSONResponse(
            content={"error": "Database connection not available."}, status_code=503
        )

    valid_sort_by = {
        "time": "created_at",
        "views": "view_count",
        "alpha": "short_description",
    }
    valid_order = {"asc": True, "desc": False}
    sort_column = valid_sort_by.get(sort_by, "created_at")
    ascending = valid_order.get(order, False)
    descending = not ascending  # Calculate descending flag

    try:
        # Query the `prompts` table
        response = (
            supabase.table("prompts")
            .select(
                "short_description, created_at, view_count", count=CountMethod.exact
            )  # Keep count="exact" for now, monitor Supabase docs if needed
            .order(
                sort_column, desc=descending
            )  # Use desc parameter instead of ascending
            .execute()
        )

        logger.info(f"Fetched {response.count} prompts from database.")
        prompts_data = []
        if response.data:
            for row in response.data:
                created_at_iso = (
                    row["created_at"].isoformat() if row.get("created_at") else None
                )
                prompts_data.append(
                    {
                        "prompt": row.get("short_description"),  # Use short_description
                        "created_at": created_at_iso,
                        "view_count": row.get("view_count"),
                    }
                )
            return JSONResponse(content=prompts_data)
        else:
            return JSONResponse(content=[])

    except Exception as e:
        logger.exception("Error fetching prompts from Supabase", exc_info=e)
        return JSONResponse(
            content={"error": "Failed to fetch prompts."}, status_code=500
        )


# --- Health Check ---
@app.get("/health")
async def health_check():
    logger.debug("Health check requested.")
    db_status = "ok" if supabase else "unavailable"
    llm_status = "ok" if anthropic_client else "unavailable"
    return {"status": "ok", "database": db_status, "llm_service": llm_status}
