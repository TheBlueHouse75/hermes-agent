#!/usr/bin/env python3
"""
Image Generation Tools Module

This module provides image generation tools using FAL.ai's FLUX 2 Pro model with 
automatic upscaling via FAL.ai's Clarity Upscaler for enhanced image quality.

Available tools:
- image_generate_tool: Generate images from text prompts with automatic upscaling

Features:
- High-quality image generation using FLUX 2 Pro model
- Automatic 2x upscaling using Clarity Upscaler for enhanced quality
- Comprehensive parameter control (size, steps, guidance, etc.)
- Proper error handling and validation with fallback to original images
- Debug logging support
- Sync mode for immediate results

Usage:
    from image_generation_tool import image_generate_tool
    import asyncio
    
    # Generate and automatically upscale an image
    result = await image_generate_tool(
        prompt="A serene mountain landscape with cherry blossoms",
        image_size="landscape_4_3",
        num_images=1
    )
"""

import base64
import json
import logging
import os
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import fal_client
from hermes_cli.env_loader import load_hermes_dotenv
from tools.debug_helpers import DebugSession

logger = logging.getLogger(__name__)


def _refresh_image_env() -> None:
    """Load ~/.hermes/.env on demand so long-lived runtimes see updated keys."""
    load_hermes_dotenv()


# Configuration for image generation
DEFAULT_MODEL = "fal-ai/flux-2-pro"
DEFAULT_OPENAI_MODEL = "gpt-image-1.5"
DEFAULT_ASPECT_RATIO = "landscape"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_NUM_IMAGES = 1
DEFAULT_OUTPUT_FORMAT = "png"

# Safety settings
ENABLE_SAFETY_CHECKER = False
SAFETY_TOLERANCE = "5"  # Maximum tolerance (1-5, where 5 is most permissive)

# Aspect ratio mapping - simplified choices for model to select
ASPECT_RATIO_MAP = {
    "landscape": "landscape_16_9",
    "square": "square_hd",
    "portrait": "portrait_16_9"
}
OPENAI_SIZE_MAP = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}
VALID_ASPECT_RATIOS = list(ASPECT_RATIO_MAP.keys())

# Configuration for automatic upscaling
UPSCALER_MODEL = "fal-ai/clarity-upscaler"
UPSCALER_FACTOR = 2
UPSCALER_SAFETY_CHECKER = False
UPSCALER_DEFAULT_PROMPT = "masterpiece, best quality, highres"
UPSCALER_NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:2)"
UPSCALER_CREATIVITY = 0.35
UPSCALER_RESEMBLANCE = 0.6
UPSCALER_GUIDANCE_SCALE = 4
UPSCALER_NUM_INFERENCE_STEPS = 18

# Valid parameter values for validation based on FLUX 2 Pro documentation
VALID_IMAGE_SIZES = [
    "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
]
VALID_OUTPUT_FORMATS = ["jpeg", "png"]
VALID_ACCELERATION_MODES = ["none", "regular", "high"]

_debug = DebugSession("image_tools", env_var="IMAGE_TOOLS_DEBUG")


def _clean_env_secret(name: str) -> str:
    """Return a sanitized secret value, treating control-char garbage as unset."""
    raw = os.getenv(name, "")
    if not raw:
        return ""
    cleaned = "".join(ch for ch in raw.strip() if ch.isprintable())
    return cleaned.strip()


def _validate_parameters(
    image_size: Union[str, Dict[str, int]], 
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    output_format: str,
    acceleration: str = "none"
) -> Dict[str, Any]:
    """
    Validate and normalize image generation parameters for FLUX 2 Pro model.
    
    Args:
        image_size: Either a preset string or custom size dict
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale value
        num_images: Number of images to generate
        output_format: Output format for images
        acceleration: Acceleration mode for generation speed
    
    Returns:
        Dict[str, Any]: Validated and normalized parameters
    
    Raises:
        ValueError: If any parameter is invalid
    """
    validated = {}
    
    # Validate image_size
    if isinstance(image_size, str):
        if image_size not in VALID_IMAGE_SIZES:
            raise ValueError(f"Invalid image_size '{image_size}'. Must be one of: {VALID_IMAGE_SIZES}")
        validated["image_size"] = image_size
    elif isinstance(image_size, dict):
        if "width" not in image_size or "height" not in image_size:
            raise ValueError("Custom image_size must contain 'width' and 'height' keys")
        if not isinstance(image_size["width"], int) or not isinstance(image_size["height"], int):
            raise ValueError("Custom image_size width and height must be integers")
        if image_size["width"] < 64 or image_size["height"] < 64:
            raise ValueError("Custom image_size dimensions must be at least 64x64")
        if image_size["width"] > 2048 or image_size["height"] > 2048:
            raise ValueError("Custom image_size dimensions must not exceed 2048x2048")
        validated["image_size"] = image_size
    else:
        raise ValueError("image_size must be either a preset string or a dict with width/height")
    
    # Validate num_inference_steps
    if not isinstance(num_inference_steps, int) or num_inference_steps < 1 or num_inference_steps > 100:
        raise ValueError("num_inference_steps must be an integer between 1 and 100")
    validated["num_inference_steps"] = num_inference_steps
    
    # Validate guidance_scale (FLUX 2 Pro default is 4.5)
    if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0.1 or guidance_scale > 20.0:
        raise ValueError("guidance_scale must be a number between 0.1 and 20.0")
    validated["guidance_scale"] = float(guidance_scale)
    
    # Validate num_images
    if not isinstance(num_images, int) or num_images < 1 or num_images > 4:
        raise ValueError("num_images must be an integer between 1 and 4")
    validated["num_images"] = num_images
    
    # Validate output_format
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"Invalid output_format '{output_format}'. Must be one of: {VALID_OUTPUT_FORMATS}")
    validated["output_format"] = output_format
    
    # Validate acceleration
    if acceleration not in VALID_ACCELERATION_MODES:
        raise ValueError(f"Invalid acceleration '{acceleration}'. Must be one of: {VALID_ACCELERATION_MODES}")
    validated["acceleration"] = acceleration
    
    return validated


def _upscale_image(image_url: str, original_prompt: str) -> Dict[str, Any]:
    """
    Upscale an image using FAL.ai's Clarity Upscaler.
    
    Uses the synchronous fal_client API to avoid event loop lifecycle issues
    when called from threaded contexts (e.g. gateway thread pool).
    
    Args:
        image_url (str): URL of the image to upscale
        original_prompt (str): Original prompt used to generate the image
    
    Returns:
        Dict[str, Any]: Upscaled image data or None if upscaling fails
    """
    try:
        logger.info("Upscaling image with Clarity Upscaler...")
        
        # Prepare arguments for upscaler
        upscaler_arguments = {
            "image_url": image_url,
            "prompt": f"{UPSCALER_DEFAULT_PROMPT}, {original_prompt}",
            "upscale_factor": UPSCALER_FACTOR,
            "negative_prompt": UPSCALER_NEGATIVE_PROMPT,
            "creativity": UPSCALER_CREATIVITY,
            "resemblance": UPSCALER_RESEMBLANCE,
            "guidance_scale": UPSCALER_GUIDANCE_SCALE,
            "num_inference_steps": UPSCALER_NUM_INFERENCE_STEPS,
            "enable_safety_checker": UPSCALER_SAFETY_CHECKER
        }
        
        # Use sync API — fal_client.submit() uses httpx.Client (no event loop).
        # The async API (submit_async) caches a global httpx.AsyncClient via
        # @cached_property, which breaks when asyncio.run() destroys the loop
        # between calls (gateway thread-pool pattern).
        handler = fal_client.submit(
            UPSCALER_MODEL,
            arguments=upscaler_arguments
        )
        
        # Get the upscaled result (sync — blocks until done)
        result = handler.get()
        
        if result and "image" in result:
            upscaled_image = result["image"]
            logger.info("Image upscaled successfully to %sx%s", upscaled_image.get('width', 'unknown'), upscaled_image.get('height', 'unknown'))
            return {
                "url": upscaled_image["url"],
                "width": upscaled_image.get("width", 0),
                "height": upscaled_image.get("height", 0),
                "upscaled": True,
                "upscale_factor": UPSCALER_FACTOR
            }
        else:
            logger.error("Upscaler returned invalid response")
            return None
            
    except Exception as e:
        logger.error("Error upscaling image: %s", e, exc_info=True)
        return None


def check_openai_api_key() -> bool:
    """Check if an OpenAI-compatible API key is available."""
    _refresh_image_env()
    return bool(_clean_env_secret("OPENAI_API_KEY"))



def _get_available_backend() -> Optional[str]:
    """Return the preferred image backend available in the current environment."""
    if check_fal_api_key():
        return "fal"
    if check_openai_api_key():
        return "openai"
    return None



def _get_generated_images_dir() -> Path:
    hermes_home = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))
    output_dir = hermes_home / "generated-images"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir



def _save_base64_image(image_b64: str, output_format: str) -> str:
    suffix = ".png" if output_format == "png" else ".jpg"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = _get_generated_images_dir() / f"image-{timestamp}{suffix}"
    output_path.write_bytes(base64.b64decode(image_b64))
    return str(output_path)



def _generate_with_openai(
    prompt: str,
    aspect_ratio: str,
    output_format: str,
) -> Dict[str, Any]:
    from openai import OpenAI

    _refresh_image_env()
    client_kwargs = {"api_key": _clean_env_secret("OPENAI_API_KEY")}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    response = client.images.generate(
        model=os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_OPENAI_MODEL),
        prompt=prompt,
        size=OPENAI_SIZE_MAP[aspect_ratio],
        quality="high",
    )

    data = getattr(response, "data", None) or []
    if not data:
        raise ValueError("Invalid response from OpenAI image API - no images returned")

    first_image = data[0]
    image_url = getattr(first_image, "url", None) or (first_image.get("url") if isinstance(first_image, dict) else None)
    if image_url:
        return {"url": image_url, "upscaled": False, "backend": "openai"}

    image_b64 = getattr(first_image, "b64_json", None) or (first_image.get("b64_json") if isinstance(first_image, dict) else None)
    if image_b64:
        return {
            "url": _save_base64_image(image_b64, output_format),
            "upscaled": False,
            "backend": "openai",
        }

    raise ValueError("Invalid response from OpenAI image API - missing image payload")



def _generate_with_fal(
    prompt: str,
    aspect_ratio_lower: str,
    image_size: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    output_format: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    validated_params = _validate_parameters(
        image_size, num_inference_steps, guidance_scale, num_images, output_format, "none"
    )

    arguments = {
        "prompt": prompt.strip(),
        "image_size": validated_params["image_size"],
        "num_inference_steps": validated_params["num_inference_steps"],
        "guidance_scale": validated_params["guidance_scale"],
        "num_images": validated_params["num_images"],
        "output_format": validated_params["output_format"],
        "enable_safety_checker": ENABLE_SAFETY_CHECKER,
        "safety_tolerance": SAFETY_TOLERANCE,
        "sync_mode": True,
    }

    if seed is not None and isinstance(seed, int):
        arguments["seed"] = seed

    logger.info("Submitting generation request to FAL.ai FLUX 2 Pro...")
    logger.info("  Model: %s", DEFAULT_MODEL)
    logger.info("  Aspect Ratio: %s -> %s", aspect_ratio_lower, image_size)
    logger.info("  Steps: %s", validated_params["num_inference_steps"])
    logger.info("  Guidance: %s", validated_params["guidance_scale"])

    handler = fal_client.submit(
        DEFAULT_MODEL,
        arguments=arguments
    )
    result = handler.get()

    if not result or "images" not in result:
        raise ValueError("Invalid response from FAL.ai API - no images returned")

    images = result.get("images", [])
    if not images:
        raise ValueError("No images were generated")

    formatted_images = []
    for img in images:
        if isinstance(img, dict) and "url" in img:
            original_image = {
                "url": img["url"],
                "width": img.get("width", 0),
                "height": img.get("height", 0)
            }

            upscaled_image = _upscale_image(img["url"], prompt.strip())
            if upscaled_image:
                formatted_images.append(upscaled_image)
            else:
                logger.warning("Using original image as fallback")
                original_image["upscaled"] = False
                formatted_images.append(original_image)

    if not formatted_images:
        raise ValueError("No valid image URLs returned from API")

    return {
        "url": formatted_images[0]["url"],
        "count": len(formatted_images),
        "upscaled_count": sum(1 for img in formatted_images if img.get("upscaled", False)),
        "backend": "fal",
    }


def image_generate_tool(
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_images: int = DEFAULT_NUM_IMAGES,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    seed: Optional[int] = None
) -> str:
    """
    Generate images from text prompts using FAL.ai's FLUX 2 Pro model with automatic upscaling.
    
    Uses the synchronous fal_client API to avoid event loop lifecycle issues.
    The async API's global httpx.AsyncClient (cached via @cached_property) breaks
    when asyncio.run() destroys and recreates event loops between calls, which
    happens in the gateway's thread-pool pattern.
    
    Args:
        prompt (str): The text prompt describing the desired image
        aspect_ratio (str): Image aspect ratio - "landscape", "square", or "portrait" (default: "landscape")
        num_inference_steps (int): Number of denoising steps (1-50, default: 50)
        guidance_scale (float): How closely to follow prompt (0.1-20.0, default: 4.5)
        num_images (int): Number of images to generate (1-4, default: 1)
        output_format (str): Image format "jpeg" or "png" (default: "png")
        seed (Optional[int]): Random seed for reproducible results (optional)
    
    Returns:
        str: JSON string containing minimal generation results:
             {
                 "success": bool,
                 "image": str or None  # URL of the upscaled image, or None if failed
             }
    """
    # Validate and map aspect_ratio to actual image_size
    aspect_ratio_lower = aspect_ratio.lower().strip() if aspect_ratio else DEFAULT_ASPECT_RATIO
    if aspect_ratio_lower not in ASPECT_RATIO_MAP:
        logger.warning("Invalid aspect_ratio '%s', defaulting to '%s'", aspect_ratio, DEFAULT_ASPECT_RATIO)
        aspect_ratio_lower = DEFAULT_ASPECT_RATIO
    image_size = ASPECT_RATIO_MAP[aspect_ratio_lower]
    
    debug_call_data = {
        "parameters": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "output_format": output_format,
            "seed": seed
        },
        "error": None,
        "success": False,
        "images_generated": 0,
        "generation_time": 0
    }
    
    start_time = datetime.datetime.now()
    
    try:
        logger.info("Generating %s image(s) with FLUX 2 Pro: %s", num_images, prompt[:80])
        
        # Validate prompt
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt is required and must be a non-empty string")
        
        backend = _get_available_backend()
        if not backend:
            raise ValueError("Neither FAL_KEY nor OPENAI_API_KEY environment variable is set")

        logger.info("Using image backend: %s", backend)

        if backend == "fal":
            generation_result = _generate_with_fal(
                prompt=prompt,
                aspect_ratio_lower=aspect_ratio_lower,
                image_size=image_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images=num_images,
                output_format=output_format,
                seed=seed,
            )
        else:
            generation_result = _generate_with_openai(
                prompt=prompt.strip(),
                aspect_ratio=aspect_ratio_lower,
                output_format=output_format,
            )

        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        generated_count = generation_result.get("count", 1)
        upscaled_count = generation_result.get("upscaled_count", 0)
        logger.info(
            "Generated %s image(s) in %.1fs via %s (%s upscaled)",
            generated_count,
            generation_time,
            generation_result.get("backend", backend),
            upscaled_count,
        )

        # Prepare successful response - minimal format
        response_data = {
            "success": True,
            "image": generation_result["url"]
        }
        
        debug_call_data["success"] = True
        debug_call_data["images_generated"] = generated_count
        debug_call_data["generation_time"] = generation_time
        
        # Log debug information
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        error_msg = f"Error generating image: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        
        # Prepare error response - minimal format
        response_data = {
            "success": False,
            "image": None
        }
        
        debug_call_data["error"] = error_msg
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)


def check_fal_api_key() -> bool:
    """
    Check if the FAL.ai API key is available in environment variables.
    
    Returns:
        bool: True if FAL_KEY is available, False otherwise
    """
    _refresh_image_env()
    return bool(_clean_env_secret("FAL_KEY"))


def check_image_generation_requirements() -> bool:
    """
    Check if all requirements for image generation tools are met.
    
    Returns:
        bool: True if requirements are met, False otherwise
    """
    try:
        backend = _get_available_backend()
        if backend == "fal":
            import fal_client
            return True
        if backend == "openai":
            import openai
            return True
        return False
        
    except ImportError:
        return False


def get_debug_session_info() -> Dict[str, Any]:
    """
    Get information about the current debug session.
    
    Returns:
        Dict[str, Any]: Dictionary containing debug session information
    """
    return _debug.get_session_info()


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🎨 Image Generation Tools Module - FLUX 2 Pro + Auto Upscaling")
    print("=" * 60)
    
    # Check if API key is available
    api_available = check_fal_api_key()
    
    if not api_available:
        print("❌ FAL_KEY environment variable not set")
        print("Please set your API key: export FAL_KEY='your-key-here'")
        print("Get API key at: https://fal.ai/")
        exit(1)
    else:
        print("✅ FAL.ai API key found")
    
    # Check if fal_client is available
    try:
        import fal_client
        print("✅ fal_client library available")
    except ImportError:
        print("❌ fal_client library not found")
        print("Please install: pip install fal-client")
        exit(1)
    
    print("🛠️ Image generation tools ready for use!")
    print(f"🤖 Using model: {DEFAULT_MODEL}")
    print(f"🔍 Auto-upscaling with: {UPSCALER_MODEL} ({UPSCALER_FACTOR}x)")
    
    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: ./logs/image_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set IMAGE_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from image_generation_tool import image_generate_tool")
    print("  import asyncio")
    print("")
    print("  async def main():")
    print("      # Generate image with automatic 2x upscaling")
    print("      result = await image_generate_tool(")
    print("          prompt='A serene mountain landscape with cherry blossoms',")
    print("          image_size='landscape_4_3',")
    print("          num_images=1")
    print("      )")
    print("      print(result)")
    print("  asyncio.run(main())")
    
    print("\nSupported image sizes:")
    for size in VALID_IMAGE_SIZES:
        print(f"  - {size}")
    print("  - Custom: {'width': 512, 'height': 768} (if needed)")
    
    print("\nAcceleration modes:")
    for mode in VALID_ACCELERATION_MODES:
        print(f"  - {mode}")
    
    print("\nExample prompts:")
    print("  - 'A candid street photo of a woman with a pink bob and bold eyeliner'")
    print("  - 'Modern architecture building with glass facade, sunset lighting'")
    print("  - 'Abstract art with vibrant colors and geometric patterns'")
    print("  - 'Portrait of a wise old owl perched on ancient tree branch'")
    print("  - 'Futuristic cityscape with flying cars and neon lights'")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export IMAGE_TOOLS_DEBUG=true")
    print("  # Debug logs capture all image generation calls and results")
    print("  # Logs saved to: ./logs/image_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

IMAGE_GENERATE_SCHEMA = {
    "name": "image_generate",
    "description": "Generate high-quality images from text prompts. Uses FLUX 2 Pro with automatic 2x upscaling when FAL is configured, otherwise falls back to OpenAI image generation when an OpenAI API key is available. Returns a single image URL or absolute local file path. For local files, send it with MEDIA:/absolute/path.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt describing the desired image. Be detailed and descriptive."
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["landscape", "square", "portrait"],
                "description": "The aspect ratio of the generated image. 'landscape' is 16:9 wide, 'portrait' is 16:9 tall, 'square' is 1:1.",
                "default": "landscape"
            }
        },
        "required": ["prompt"]
    }
}


def _handle_image_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "prompt is required for image generation"})
    return image_generate_tool(
        prompt=prompt,
        aspect_ratio=args.get("aspect_ratio", "landscape"),
        num_inference_steps=50,
        guidance_scale=4.5,
        num_images=1,
        output_format="png",
        seed=None,
    )


registry.register(
    name="image_generate",
    toolset="image_gen",
    schema=IMAGE_GENERATE_SCHEMA,
    handler=_handle_image_generate,
    check_fn=check_image_generation_requirements,
    requires_env=["FAL_KEY", "OPENAI_API_KEY"],
    is_async=False,  # Switched to sync fal_client API to fix "Event loop is closed" in gateway
    emoji="🎨",
)
