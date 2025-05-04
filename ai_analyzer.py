#!/usr/bin/env python3
import os
import sys
import requests
from dotenv import load_dotenv
import time
import logging
from logging.handlers import RotatingFileHandler


# Set up logging
# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "ai_analyzer.log")

# Configure logger to write to both file and console
logger = logging.getLogger("ai_analyzer")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
console_handler = logging.StreamHandler()

# Create formatters and add to handlers
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
file_formatter = logging.Formatter(log_format)
console_formatter = logging.Formatter(log_format)
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

logger.info(f"AI Analyzer initialized. Endpoint: {azure_endpoint}")
logger.info(f"API key available: {bool(api_key)}")


class CarAIAnalyzer:
    """
    Class for analyzing car listings using Azure OpenAI API
    """
    
    def __init__(self, logger=None):
        """
        Initialize the analyzer
        
        Args:
            logger: Optional logger instance to use instead of creating a new one
        """
        # Set up logging - either use the provided logger or create a new one
        self.logger = logger or logging.getLogger("ai_analyzer")
        
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        
        self.logger.info(f"AI Analyzer initialized. Endpoint: {self.azure_endpoint}")
        self.logger.info(f"API key available: {bool(self.api_key)}")
        
        # Check if API key is available
        self.is_available = bool(self.api_key) and bool(self.azure_endpoint)
    
    def analyze_car(self, car_data):
        """
        Analyze a single car listing and return the AI's assessment
        
        Args:
            car_data (dict): Dictionary containing car listing data
            
        Returns:
            str: AI analysis of the car or error message
        """
        self.logger.info(f"analyze_car called for: {car_data.get('title', 'unknown')}")
        
        if not self.is_available:
            self.logger.error("API configuration missing. Cannot perform analysis.")
            return "Azure OpenAI API key not found. Cannot perform analysis."
        
        # Track retry attempts
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                self.logger.info(f"Analysis attempt #{retry_count + 1} of {max_retries + 1}")
                
                # Extract essential car info
                car_info = {
                    "Title": car_data.get('title', 'N/A'),
                    "Price": car_data.get('Totalpris', 'N/A'),
                    "Mileage": car_data.get('Kilometerstand', 'N/A'),
                    "Year": car_data.get('Modellår', 'N/A')
                }
                
                # Add optional fields if they exist
                optional_fields = [
                    'Girkasse', 'Drivstoff', 'Motor', 'Effekt', 'Hjuldrift'
                ]
                for field in optional_fields:
                    if field in car_data and car_data[field]:
                        car_info[field] = car_data[field]
                
                self.logger.info(f"Prepared car info: {car_info}")
                
                # Create prompt
                prompt_parts = ["Analyze this car listing:"]
                # Add formatted basic info first
                for key, value in car_info.items():
                    prompt_parts.append(f"{key}: {value}")
                
                # Add all available scraped data for better information extraction
                prompt_parts.append("\nFull listing data:")
                for key, value in car_data.items():
                    if key not in car_info and value:
                        prompt_parts.append(f"{key}: {value}")
                
                prompt_parts.append(
                    "\nThe scraped information is from finn.no car listings. "
                    "Please analyze this car from the perspective of a potential buyer:\n"
                    "1. Mention notable features, equipment, and selling points\n"
                    "2. Assess if the price appears reasonable given the specs\n"
                    "3. Note any potential issues or concerns\n"
                    "4. Give a brief overall assessment\n\n"
                    "Format your response as clear bullet points. "
                    "Keep it concise, under 120 words."
                )
                prompt_text = "\n".join(prompt_parts)
                
                self.logger.info(f"Constructed prompt with {len(prompt_text)} characters")
                
                # Make the API request to Azure OpenAI
                url = (f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/"
                       f"chat/completions?api-version={self.azure_api_version}")
                
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.api_key
                }
                
                payload = {
                    "messages": [
                        {"role": "system", "content": (
                            "You are an experienced car expert and buyer's consultant with deep "
                            "knowledge of the Norwegian car market. Your job is to provide concise, "
                            "balanced, and helpful assessments of car listings to help buyers "
                            "make informed decisions."
                        )},
                        {"role": "user", "content": prompt_text}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 250
                }
                
                self.logger.info(f"Making API request to endpoint: {url[:60]}...")
                
                # Add timeout to prevent hanging indefinitely - use a longer timeout
                start_time = time.time()
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=45  # Increased timeout to 45 seconds
                )
                request_time = time.time() - start_time
                self.logger.info(f"API request completed in {request_time:.2f} seconds with status code {response.status_code}")
                
                # Check if request was successful
                response.raise_for_status()
                
                # Parse and return the response
                response_data = response.json()
                if "choices" in response_data and response_data["choices"]:
                    result = response_data["choices"][0]["message"]["content"]
                    self.logger.info(f"Successfully extracted response content: {result[:50]}...")
                    return result
                else:
                    self.logger.error(f"API response missing expected data: {response_data}")
                    return ("No analysis could be generated. "
                            "API response did not contain expected data.")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {retry_count + 1} of {max_retries + 1}")
                retry_count += 1
                if retry_count <= max_retries:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** retry_count
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                self.logger.error("All retry attempts failed due to timeout")
                return ("API request timed out after multiple attempts. "
                        "The service may be experiencing high load. "
                        "Please try again later.")
                
            except requests.exceptions.ConnectionError as conn_err:
                self.logger.warning(f"Connection error on attempt {retry_count + 1}: {str(conn_err)}")
                retry_count += 1
                if retry_count <= max_retries:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** retry_count
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                self.logger.error("All retry attempts failed due to connection errors")
                return ("Connection error after multiple attempts. "
                        "Please check your internet connection and try again.")
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request exception: {str(e)}", exc_info=True)
                return f"API request error: {str(e)}"
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                return f"Error analyzing car: {str(e)}"


# Example code for testing when running this file directly
if __name__ == "__main__":
    example = {
        "title": "Volvo XC90 Momentum",
        "Totalpris": "450000",
        "Kilometerstand": "30000",
        "Modellår": "2019",
        "Girkasse": "Automat",
        "Drivstoff": "Diesel",
        "Motor": "2.0L",
        "Effekt": "190 hk",
        "Hjuldrift": "Forhjulsdrift",
    }

    try:
        analyzer = CarAIAnalyzer()
        if analyzer.is_available:
            result = analyzer.analyze_car(example)
            print("\nAI Analysis:\n", result)
        else:
            print("Azure OpenAI API configuration not found. Please check your .env file.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
