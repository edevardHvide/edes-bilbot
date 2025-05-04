#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

# Check required configuration
if not api_key:
    print("Error: AZURE_OPENAI_API_KEY not found in environment or .env file")
    exit(1)
if not azure_endpoint:
    print("Error: AZURE_OPENAI_ENDPOINT not found in environment or .env file")
    exit(1)

# Print configuration (with masked API key for security)
print("\n=== Configuration ===")
print(f"Endpoint: {azure_endpoint}")
print(f"API Version: {azure_api_version}")
print(f"Deployment: {azure_deployment}")
masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "****"
print(f"API Key: {masked_key}")
print("====================\n")

# Simple function to send a request to Azure OpenAI API
def ask_azure_openai(prompt):
    """Send a text prompt to Azure OpenAI and return the response"""
    
    url = (f"{azure_endpoint}/openai/deployments/{azure_deployment}/"
           f"chat/completions?api-version={azure_api_version}")
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    print(f"Sending request to: {url}")
    response = requests.post(
        url,
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        return f"Error: {response.status_code} - {response.text}"

# Test with a simple question
prompt = "What are the top 3 things to consider when buying a car?"
print(f"Asking Azure OpenAI: {prompt}")

try:
    response = ask_azure_openai(prompt)
    
    if isinstance(response, dict) and "choices" in response:
        answer = response["choices"][0]["message"]["content"]
        print("\nResponse from Azure OpenAI:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
    else:
        print(f"Unexpected response format: {response}")
except Exception as e:
    print(f"Error making request: {e}")

# Function to analyze car listings
def analyze_car(car_info):
    """Analyze a car listing using Azure OpenAI and return insights"""
    
    car_prompt = f"""
Analyze this car listing:
Title: {car_info.get('title', 'N/A')}
Price: {car_info.get('price', 'N/A')}
Year: {car_info.get('year', 'N/A')}
Mileage: {car_info.get('mileage', 'N/A')}

What are the key features and potential issues to be aware of with this car?
Keep it short and in bullet points.
"""
    response = ask_azure_openai(car_prompt)
    if isinstance(response, dict) and "choices" in response:
        return response["choices"][0]["message"]["content"]
    else:
        return f"Error analyzing car: {response}"

# Example of using analyze_car:
print("\nAnalyzing sample car...")
sample_car = {
    "title": "BMW X5 xDrive40i 2022",
    "price": "1,199,000 kr",
    "year": "2022",
    "mileage": "15,000 km"
}
car_analysis = analyze_car(sample_car)
print(car_analysis) 