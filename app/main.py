import os
import asyncio
import logging
from typing import Optional, Dict, Any  # Import Dict and Any
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from supabase import Client, create_client  # type: ignore
from postgrest.types import CountMethod
from dotenv import load_dotenv
import json
from pydantic import BaseModel
from datetime import datetime  # Add datetime import
import yaml  # Import YAML

# --- Import Anthropic ---
from anthropic import AsyncAnthropic, APIError

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

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

# --- Load Configuration ---
config: Dict[str, Any] = {}  # Global config dictionary
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, "config.yaml")
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        if not config or "llm" not in config:
            raise ValueError("Invalid config structure: 'llm' section missing.")
        logger.info(f"Successfully loaded configuration from {config_path}")
except FileNotFoundError:
    logger.error(
        f"Configuration file not found at {config_path}. LLM functionality may be limited."
    )
    # Optionally set default values or exit
    config = {"llm": {}}  # Ensure config['llm'] exists
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file {config_path}: {e}")
    config = {"llm": {}}  # Ensure config['llm'] exists
except Exception as e:
    logger.exception(
        f"An unexpected error occurred while loading configuration from {config_path}",
        exc_info=e,
    )
    config = {"llm": {}}  # Ensure config['llm'] exists

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


async def get_or_create_model_params(
    model_name: str, system_prompt: str, max_tokens: int, upsert_first: bool = False
) -> str:
    """Finds existing model parameters or creates them, returning the params_id."""
    if not supabase:
        logger.error("Supabase client not available for get_or_create_model_params")
        raise HTTPException(
            status_code=503, detail="Database connection not available."
        )

    params_to_find_or_insert = {
        "model_name": model_name,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
    }
    # Define the columns that form the unique constraint for conflict resolution
    conflict_columns = "model_name, system_prompt, max_tokens"

    if not upsert_first:
        try:
            select_resp = (
                supabase.table("model_params")
                .select("params_id")
                .match(params_to_find_or_insert)
                .limit(1)
                .execute()
            )
            if select_resp.data:
                params_id = select_resp.data[0]["params_id"]
                return params_id
        except Exception as e:
            logger.warning("Error during model_params select", exc_info=e)
        logger.warning(
            f"Could not find model_params: {params_to_find_or_insert}. Creating new one."
        )

    logger.info(f"Upserting model_params: {params_to_find_or_insert}")
    upsert_result = None
    try:
        upsert_result = (
            supabase.table("model_params")
            .upsert(
                params_to_find_or_insert,
                on_conflict=conflict_columns,
                returning="representation",  # type: ignore
                ignore_duplicates=False,  # Ensure we get the existing row if conflict
            )
            .execute()
        )

        if upsert_result.data and len(upsert_result.data) > 0:
            params_id = upsert_result.data[0]["params_id"]
            logger.info(f"Found or created model_params with ID: {params_id}")
            return params_id
    except Exception as e:
        logger.exception("Error during model_params upsert", exc_info=e)
    logger.error(
        f"Upsert failed or did not return data for model_params: {params_to_find_or_insert}. Result: {upsert_result}"
    )
    return handle_model_params_error(params_to_find_or_insert)


def handle_model_params_error(params_to_find_or_insert):
    """Handle errors in model_params operations with helpful debugging SQL."""
    # Suggest SQL that could be run manually to debug/fix the issue
    suggested_sql = f"""
-- Check if the record exists:
SELECT * FROM model_params
WHERE model_name = '{params_to_find_or_insert['model_name']}'
AND system_prompt = '{params_to_find_or_insert['system_prompt']}'
AND max_tokens = {params_to_find_or_insert['max_tokens']};

-- If not found, try inserting manually:
INSERT INTO model_params (model_name, system_prompt, max_tokens)
VALUES ('{params_to_find_or_insert['model_name']}',
        '{params_to_find_or_insert['system_prompt']}',
        {params_to_find_or_insert['max_tokens']})
RETURNING params_id;
"""
    logger.error(f"Suggested SQL to run manually: {suggested_sql}")
    raise HTTPException(
        status_code=500, detail="Failed to get or create model parameters."
    )


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
    params_id: str,
    model_name: str,
    system_prompt: str,
    messages: list,
    max_tokens: int,
):
    """
    Calls LLM stream, yields chunks, saves the full response to `responses`
    linking prompt_id and params_id, and records the initial view in `view_counts`.
    """
    full_response = ""
    error_occurred = False
    new_response_id = None

    try:
        # Pass received arguments directly to the LLM stream function
        async for chunk in get_llm_stream(
            model_name, system_prompt, messages, max_tokens
        ):
            if isinstance(chunk, str) and chunk.startswith('data: {"error":'):
                yield chunk
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
        if supabase and full_response:
            logger.info(
                f"Attempting to save new response for prompt_id: '{prompt_id}', params_id: '{params_id}'"
            )
            try:
                response_insert_result = (
                    supabase.table("responses")
                    .insert(
                        {
                            "prompt_id": prompt_id,
                            "params_id": params_id,
                            "response_text": full_response,
                        },
                        returning="representation",  # type: ignore
                    )
                    .execute()
                )

                if response_insert_result.data and len(response_insert_result.data) > 0:
                    inserted_row = response_insert_result.data[0]
                    if "response_id" in inserted_row:
                        new_response_id = inserted_row["response_id"]
                        logger.info(
                            f"Successfully saved new response (ID: {new_response_id}) for prompt_id: '{prompt_id}'"
                        )
                    else:
                        logger.error(
                            f"'response_id' not found in returned data for prompt {prompt_id}"
                        )
                        error_occurred = True
                else:
                    logger.error(
                        f"Failed to insert response or get representation for prompt {prompt_id}. Result: {response_insert_result}"
                    )
                    error_occurred = True

            except Exception as e:
                logger.exception(
                    f"Error saving new response to Supabase for prompt_id '{prompt_id}'",
                    exc_info=e,
                )
                yield f"data: {json.dumps({'error': 'Failed to save new response.'})}\n\n"
                error_occurred = True

        # --- Record the initial view in `view_counts` --- #
        if supabase and new_response_id and not error_occurred:
            try:
                logger.info(
                    f"Recording initial view for response_id: {new_response_id}"
                )
                supabase.table("view_counts").insert(
                    {"response_id": new_response_id}
                ).execute()
                logger.info(
                    f"Successfully recorded initial view for response_id: {new_response_id}"
                )
            except Exception as e:
                logger.exception(
                    f"Failed to record initial view for response_id {new_response_id}",
                    exc_info=e,
                )

        # --- Send End Event --- #
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

    # --- Determine Model Parameters from Config --- #
    llm_config = config.get("llm", {})
    model_name = llm_config.get("model_name")
    system_prompt = llm_config.get("system_prompt")
    max_tokens_str = llm_config.get("max_tokens", "1000")  # Default if missing
    prompt_template = llm_config.get(
        "prompt_template", "Write an essay about {short_description}"
    )  # Default template

    # Validate required parameters
    if not model_name:
        logger.error("LLM model_name is missing in the configuration.")
        raise HTTPException(
            status_code=500, detail="LLM configuration error: model_name missing."
        )
    if not system_prompt:
        logger.warning(
            "LLM system_prompt is missing in the configuration. Using empty system prompt."
        )
        system_prompt = ""

    # Safely convert max_tokens to int
    try:
        max_tokens = int(max_tokens_str)
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid max_tokens value '{max_tokens_str}' in config. Using default 1000."
        )
        max_tokens = 1000

    # Construct prompt_text using the template
    try:
        prompt_text = prompt_template.format(short_description=short_description)
    except KeyError:
        logger.warning(
            f"Prompt template '{prompt_template}' is missing '{{short_description}}'. Using default template."
        )
        prompt_text = f"Write an essay about {short_description}"
    except Exception as e:
        logger.error(f"Error formatting prompt template: {e}. Using basic prompt.")
        prompt_text = f"Write an essay about {short_description}"

    logger.info(f"Using LLM Parameters: model='{model_name}', max_tokens={max_tokens}")
    # logger.debug(f"System Prompt: '{system_prompt[:100]}...'") # Log truncated system prompt
    # logger.debug(f"Generated Prompt Text: '{prompt_text[:100]}...'") # Log truncated final prompt

    messages = [
        {
            "role": "user",
            "content": prompt_text,  # Use the formatted prompt text
        }
    ]
    # ------------------------------------------ #

    try:
        # --- Get or Create Model Params ID --- #
        # Pass the actual values retrieved from config
        params_id = await get_or_create_model_params(
            model_name=model_name, system_prompt=system_prompt, max_tokens=max_tokens
        )
        # ------------------------------------- #

        # --- Find existing prompt based on prompt_text (generated from template) --- #
        existing_prompt_result = (
            supabase.table("prompts")
            .select("prompt_id, created_at")
            .eq("prompt_text", prompt_text)
            .limit(1)
            .execute()
        )

        if existing_prompt_result.data and len(existing_prompt_result.data) > 0:
            # Prompt already exists, use the existing one
            prompt_info = existing_prompt_result.data[0]
            prompt_id = prompt_info["prompt_id"]
            prompt_created_at = prompt_info["created_at"]
            logger.info(f"Found existing prompt with ID: {prompt_id}")
        else:
            # Create new prompt
            new_prompt_result = (
                supabase.table("prompts")
                .insert(
                    {
                        "short_description": short_description,
                        "prompt_text": prompt_text,
                    },
                    returning="representation",  # type: ignore
                )
                .execute()
            )

            if not new_prompt_result.data or len(new_prompt_result.data) == 0:
                logger.error(
                    f"Failed to insert prompt for description: {short_description}"
                )
                raise HTTPException(status_code=500, detail="Failed to create prompt.")

            prompt_info = new_prompt_result.data[0]
            prompt_id = prompt_info["prompt_id"]
            prompt_created_at = prompt_info["created_at"]
            logger.info(f"Created new prompt with ID: {prompt_id}")

        # Determine if the prompt was newly inserted or if it already existed
        # This logic might need refinement based on exact upsert behavior / timestamps
        # A simple check: if created_at is very recent? Or compare count before/after?
        # For now, let's assume if we *found* a response below, the prompt existed.

        # --- Check for Existing Response --- #
        # Fetch the latest response for this prompt_id (regardless of params_id used to create it)
        latest_response_resp = (
            supabase.table("responses")
            .select("response_id, response_text")
            .eq("prompt_id", prompt_id)
            .order("response_created_at", desc=True)
            .limit(1)
            .execute()
        )

        if latest_response_resp.data:
            # --- Prompt Existed and has a Response --- #
            latest_response = latest_response_resp.data[0]
            latest_response_id = latest_response["response_id"]
            latest_response_text = latest_response["response_text"]
            logger.info(
                f"Found existing prompt (ID: {prompt_id}) and latest response (ID: {latest_response_id}). Streaming cached response."
            )

            # Record View
            try:
                logger.info(f"Recording view for response_id: {latest_response_id}")
                supabase.table("view_counts").insert(
                    {"response_id": latest_response_id}
                ).execute()
            except Exception as e:
                logger.exception(
                    f"Failed to record view for response_id {latest_response_id}",
                    exc_info=e,
                )

            # Stream the cached/latest response
            async def stream_latest_cached():
                # chunk_size = 20
                chunk_size = len(latest_response_text)
                for i in range(0, len(latest_response_text), chunk_size):
                    chunk = latest_response_text[i : i + chunk_size]
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0.01)
                yield f"data: {json.dumps({'end': True})}\n\n"

            return StreamingResponse(
                stream_latest_cached(), media_type="text/event-stream"
            )

        else:
            # --- Prompt was Newly Created OR Existed but has NO responses --- #
            # This happens if the upsert created the prompt, OR if the prompt existed
            # but its previous responses were deleted (or never created).
            logger.info(
                f"Prompt (ID: {prompt_id}) is new or has no existing responses. Generating new response with params_id {params_id}."
            )
            # Generate, stream, and save the first response for this prompt using current params
            return StreamingResponse(
                stream_and_save_new_response(
                    prompt_id,  # The ID from the upsert
                    params_id,  # The ID for the *current* model params
                    model_name,
                    system_prompt,
                    messages,
                    max_tokens,
                ),
                media_type="text/event-stream",
            )

    except Exception as e:
        logger.exception(
            f"Error processing /ask request for description '{short_description}'",
            exc_info=e,
        )

        async def error_stream():
            yield f"data: {json.dumps({'error': f'Server error processing request.'})}\n\n"

        return StreamingResponse(
            error_stream(), media_type="text/event-stream", status_code=500
        )


@app.get("/essays", response_class=JSONResponse)
async def get_essays():
    """Fetches the list of saved prompts and their total view counts (unsorted)."""
    logger.info("Received /essays request.")
    if not supabase:
        logger.error("Supabase client not available for /essays request.")
        return JSONResponse(
            content={"error": "Database connection not available."}, status_code=503
        )

    try:
        # --- Query Prompts and Aggregate View Counts (Using Option 2 Logic) --- #
        # Fetch all prompts first
        prompts_resp = (
            supabase.table("prompts")
            .select("prompt_id, short_description, created_at")
            .execute()
        )
        if not prompts_resp.data:
            return JSONResponse(content=[])

        prompts_map = {p["prompt_id"]: p for p in prompts_resp.data}
        prompt_ids = list(prompts_map.keys())

        # Fetch response IDs linked to these prompts
        responses_ids_resp = (
            supabase.table("responses")
            .select("response_id")
            .in_("prompt_id", prompt_ids)
            .execute()
        )
        response_ids = (
            [r["response_id"] for r in responses_ids_resp.data]
            if responses_ids_resp.data
            else []
        )

        # Fetch view counts for these response IDs
        views_resp = (
            supabase.table("view_counts")
            .select("response_id, view_id")
            .in_("response_id", response_ids)
            .execute()
        )

        views_per_response: dict[str, int] = {}  # Type hint added
        if views_resp.data:
            for view in views_resp.data:
                resp_id = view["response_id"]
                views_per_response[resp_id] = views_per_response.get(resp_id, 0) + 1

        # Fetch responses to link prompts to view counts
        responses_linking_resp = (
            supabase.table("responses")
            .select("prompt_id, response_id")
            .in_("prompt_id", prompt_ids)
            .execute()
        )

        views_per_prompt = {pid: 0 for pid in prompt_ids}
        if responses_linking_resp.data:
            for resp in responses_linking_resp.data:
                prompt_id = resp["prompt_id"]
                response_id = resp["response_id"]
                # Ensure prompt_id exists before incrementing
                if prompt_id in views_per_prompt:
                    views_per_prompt[prompt_id] += views_per_response.get(
                        response_id, 0
                    )
                else:
                    # This case might indicate an inconsistency if a response links
                    # to a prompt_id not fetched initially. Log a warning.
                    logger.warning(
                        f"Response {response_id} links to prompt {prompt_id} which was not in the initial prompt fetch."
                    )

        # Combine data
        final_data = []
        for pid, prompt_info in prompts_map.items():
            created_at_str = prompt_info.get("created_at")
            dt_obj = None
            created_at_iso = None
            if created_at_str:
                try:
                    # Handle potential 'Z' timezone indicator which Python < 3.11 doesn't parse directly
                    if created_at_str.endswith("Z"):
                        created_at_str_parsed = created_at_str[:-1] + "+00:00"
                    else:
                        created_at_str_parsed = created_at_str
                    dt_obj = datetime.fromisoformat(created_at_str_parsed)
                    created_at_iso = (
                        dt_obj.isoformat()
                    )  # Format back for JSON if needed, keeps original offset
                except ValueError:
                    logger.warning(
                        f"Could not parse created_at string: {created_at_str}. Leaving as is."
                    )
                    created_at_iso = (
                        created_at_str  # Keep original string if parse fails
                    )

            final_data.append(
                {
                    "prompt": prompt_info.get("short_description"),
                    "created_at": created_at_iso,  # Use the potentially re-formatted ISO string for consistency in JSON
                    "_created_at_dt": dt_obj,  # Internal field for sorting
                    "view_count": views_per_prompt.get(pid, 0),
                }
            )
        logger.info(f"Processed {len(final_data)} prompts with aggregated views.")
        # -------------------------------------------------- #

        # --- Remove Python Sorting --- #
        # Results are returned unsorted
        # --- Remove internal sorting key --- #
        for item in final_data:
            if "_created_at_dt" in item:
                del item["_created_at_dt"]
        # ------------------------------ #

        return JSONResponse(content=final_data)

    except Exception as e:
        logger.exception("Error fetching prompts/views from Supabase", exc_info=e)
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
