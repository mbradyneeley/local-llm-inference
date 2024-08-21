# Llama 3 70B Inference Script

This Python script allows you to run inference on the Meta Llama 3 70B Instruct model using the Replicate API. It provides a simple interface to input prompts and receive streamed responses from the model.

## Prerequisites

- Python 3.7 or higher
- A Replicate account and API token

## Setup

1. Clone this repository or download the script files.

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the same directory as the script with your Replicate API token:
   ```
   REPLICATE_API_TOKEN=your_api_token_here
   ```

   Replace `your_api_token_here` with your actual Replicate API token.

## Usage

1. Run the script:
   ```
   python llama3_inference.py
   ```

2. When prompted, enter your desired prompt for the Llama 3 70B model.

3. The script will stream the model's response, printing it to the console.

## Features

- Streams real-time responses from the Llama 3 70B model
- Handles environment variables securely
- Implements error handling and logging
- Provides a user-friendly interface for entering prompts

## Troubleshooting

- If you encounter any issues related to the API token, ensure that your `.env` file is correctly formatted and located in the same directory as the script.
- For any other errors, check the console output for error messages. The script includes logging to help identify issues.

## Contributing

Feel free to fork this repository and submit pull requests with any improvements or features you'd like to add.

## License

This project is open-source and available under the MIT License.
