# Finn Car Scraper API

A web API built with Streamlit that scrapes car listings from Finn.no and presents the data in a structured format (CSV/JSON).

## Features

- Web interface for configuring the scraper
- Extract common car attributes (year, mileage, price, etc.)
- Support for custom attributes
- Download results as CSV or JSON
- Progress tracking during scraping
- Environment variable configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/finn-car-scraper.git
cd finn-car-scraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
   - Create a `.env` file based on the `.env.example` template
   - Customize the variables as needed

## Usage

### Local Development

Run the Streamlit app locally:
```bash
streamlit run app.py
```

The app will open in your default browser at http://localhost:8501

### Deploying to Render.com

1. Push your code to GitHub
2. Create a new Web Service on Render.com
3. Connect your GitHub repository
4. Configure the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py`
   - Add environment variables in the Render dashboard under "Environment"

## How to Use

1. Enter a Finn.no search URL with your filters applied
   - Go to Finn.no, apply your filters, and copy the URL
2. Select attributes to extract from each listing
3. Set the number of listings to scrape
4. Click "Run Scraper" and wait for the results
5. View and download the data as CSV or JSON

## Environment Variables

You can configure the app behavior with these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| DEFAULT_SEARCH_URL | Initial search URL to use | Volvo search URL |
| DEFAULT_LIMIT | Default number of listings to scrape | 5 |
| DEBUG_MODE | Enable detailed logging (True/False) | False |
| LOG_FILE | Path to log file | scraper_debug.log |
| HEADLESS_MODE | Run browser in headless mode (True/False) | True |
| CHROME_ARGS | Chrome browser arguments (comma-separated) | --disable-gpu,--no-sandbox,etc |
| REQUEST_DELAY | Seconds to wait between requests | 3 |

On Render.com, set these in the Environment section of your service dashboard.

## Project Structure

- `app.py` - Streamlit web interface
- `finn_car_scraper_app.py` - Core scraper functionality
- `requirements.txt` - Project dependencies
- `.env.example` - Template for environment variables

## Dependencies

- Streamlit - Web interface
- Selenium - Web automation
- BeautifulSoup4 - HTML parsing
- Pandas - Data handling
- python-dotenv - Environment variable management

## Notes

- Use responsibly and in accordance with Finn.no's terms of service
- Rate limits may apply when scraping many listings
- For production use, consider implementing rate limiting and proxy rotation