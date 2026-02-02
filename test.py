print("Hello, World!")


# import requests

# def get_filing_data(ticker, form_type, api_key):
#     # Construct the URL for the SEC EDGAR API
#     url = f"https://api.sec.gov/{ticker}/filings/{form_type}?api_key={api_key}"
    
#     # Make the API request
#     response = requests.get(url)
#     response.raise_for_status()
    
#     # Parse the JSON response
#     data = response.json()
#     return data

# def extract_ebitda(filing_data):
#     # Navigate through the JSON structure to find EBITDA
#     for report in filing_data['facts']['us-gaap']:
#         if 'EBITDA' in report['units']:
#             ebitda_data = report['units']['EBITDA']
#             for item in ebitda_data:
#                 frame = item.get('frame', '')
#                 value = item.get('value', 0)
#                 print(f"EBITDA ({frame}): {value}")
    
# # Example usage
# ticker = 'ELAN'
# form_type = '10-K'
# api_key = 'e9f44e44efb27c053a565505cdab7ec55038a3b18f5e3e8e878bec2bb7e6305d'

# # Get the filing data
# filing_data = get_filing_data(ticker, form_type, api_key)

# # Extract and print EBITDA
# extract_ebitda(filing_data)
