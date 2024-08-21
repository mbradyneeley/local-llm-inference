import os
import sys
import logging
from dotenv import load_dotenv
import replicate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        logger.error("REPLICATE_API_TOKEN not found in .env file")
        sys.exit(1)
    return api_token

def run_llama_inference(prompt):
    """Run inference on Llama 3 70B model and stream the output."""
    try:
        for event in replicate.stream(
            "meta/meta-llama-3-70b-instruct",
            input={"prompt": prompt}
        ):
            print(str(event), end="", flush=True)
        print()  # Print a newline at the end of the output
    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def main():
    # Load environment variables
    api_token = load_environment()
    os.environ["REPLICATE_API_TOKEN"] = api_token

    # Get user input for the prompt
    user_prompt = input("Enter your prompt for Llama 3 70B: ")
    
    logger.info("Starting Llama 3 70B inference...")
    run_llama_inference(user_prompt)
    logger.info("Inference completed.")

if __name__ == "__main__":
    main()
