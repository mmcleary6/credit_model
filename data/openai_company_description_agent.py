# %%

import json
import os
import pandas as pd
from openai import OpenAI

companies = pd.read_csv("../database/companies.csv")

agent_id = "wf_69db96ae668c81909949291d7059693808963d57cd166d8e"

client = OpenAI()

companies['description'] = None

for i in range(len(companies)):

  prompt = f"""
  Give me a company description for {companies['company'][i]} 
  at the following website {companies['url'][i]}
  where it's sector is {companies['sector'][i]} and industry is {companies['sector'][i]}.

  The results should be in JSON format with the following objects. 
  Description: a text description of the company no longer than 500 words.
  Products: a list of products and or service lines the company offers. No more then 10 products/services. Each product/service should be an individual entry in the list.
  Key_People: a list of key people (examples include founders/ceo) along with a description of why they're important and what their background is. No more then 3 people. Each person should be an individual entry in the list.
  Key_Clients: a list of import client relationships. Name each client and provide their company URL. List each client individually in the list.
  """.replace("\n", "")

  prompt = f"""Generate a company profile in JSON format for the specified company and website given its sector and industry. Do not exceed the provided quantity limits for list items. Ensure you follow logical reasoning steps to analyze and extract accurate, concise information, and do not provide conclusions (such as outputting the JSON) until all information is gathered and reasoned through.

  You must:
  - Visit and extract information from {companies['url'][i]} about the company {companies['company'][i]}, which operates in the sector {companies['sector'][i]} and industry {companies['sector'][i]}.
  - Carefully analyze the website content and any available sources for accurate details.
  - Before outputting any result, think step-by-step: first list the data points you will seek and what sources you will use to infer or extract them. Double check no categories are missing. Once the analysis is complete internally, only then provide the final JSON.
  - For each section, if information cannot be found or is ambiguous, state \"Information not available\" or leave a blank string as appropriate.
  - Determine the company's overall business type: B2B (business-to-business), B2C (business-to-consumer), or D2C (direct-to-consumer). List all that apply, separated by commas, in a separate Business_Type field.
  - For Key_People, find as many key people as possible (no limit). Include their school, area of study, and LinkedIn URL if available.
  - For Key_Clients, also determine the company's sales model for each client relationship: B2B (business-to-business), B2C (business-to-consumer), or D2C (direct-to-consumer). List all that apply, separated by commas.
  - Identify all U.S. states where the company conducts business or has a presence.
  - Identify all known office locations, including state, city, zip code, and whether the office is the headquarters.
  """.replace("\n", "") + """

  Structure the output strictly in the following JSON format:
  {
    \"Description\": \"[A concise summary of the company not exceeding 500 words, covering its background, focus, sector, industry, and core strengths.]\",
    \"Business_Type\": \"[Comma-separated list of applicable types: B2B, B2C, D2C]\",
    \"Products\": [
      \"[Product/Service 1]\",
      \"[Product/Service 2]\",
      ...
    ],
    \"Key_People\": [
      {
        \"Name\": \"[Person's Name]\",
        \"Role\": \"[Title]\",
        \"Importance\": \"[Why this person is important to the company]\",
        \"Background\": \"[A brief summary of the person's experience/background]\",
        \"School\": \"[Name of school/university attended]\",
        \"Area_of_Study\": \"[Field of study or degree]\",
        \"Linkedin_URL\": \"[LinkedIn profile URL]\"
      },
      ...
    ],
    \"Key_Clients\": [
      {
        \"Client\": \"[Client Company Name]\",
        \"URL\": \"[Client Company Website]\",
        \"Sales_Model\": \"[Comma-separated list of applicable models: B2B, B2C, D2C]\"
      },
      ...
    ],
    \"States\": [
      \"[U.S. state where the company does business]\",
      ...
    ],
    \"Office_Locations\": [
      {
        \"State\": \"[State]\",
        \"City\": \"[City]\",
        \"Zip_Code\": \"[Zip Code]\",
        \"Is_HQ\": true/false
      },
      ...
    ]
  }

  - You may include up to 10 products/services, as many key people as can be found, as many key clients as can be found, and all known office locations.
  - Entries in each list should be separate and detailed as per the above schema.
  - If there is insufficient public information on an item, indicate this politely.
  - The response should contain only the JSON, with no additional commentary.

  Example Input/Output:  
  Input:  
  Company: ExampleTech  
  URL: www.exampletech.com  
  Sector: Information Technology  
  Industry: Software Development

  Reasoning (not for output, just for reference on the reasoning sequence):  
  1. Browse www.exampletech.com and company data sources for information.  
  2. Identify the company's main description and summarize in under 500 words.  
  3. List up to 10 main products/services, making entries concise.  
  4. Identify as many key people as possible, explain their importance, summarize backgrounds, and find their school, area of study, and LinkedIn URL.  
  5. Identify as many major clients as possible, naming company, URL, and sales model (B2B, B2C, D2C).  
  6. Identify all U.S. states where the company operates.  
  7. Identify all office locations with state, city, zip code, and whether it is the headquarters.

  Output (will be substantially longer/shorter with real data; just a compact placeholder here):  
  {
    \"Description\": \"ExampleTech is a leading software development firm specializing in scalable solutions for enterprise clients across multiple industries. Founded in 2010, the company excels in cloud infrastructure, AI integration, and bespoke enterprise tools.\",
    \"Business_Type\": \"B2B\",
    \"Products\": [
      \"CloudCore Platform\",
      \"AI Insight Suite\",
      \"MobilePro Framework\"
    ],
    \"Key_People\": [
      {
        \"Name\": \"Jane Doe\",
        \"Role\": \"CEO & Founder\",
        \"Importance\": \"Guided the company's strategic vision and initial growth.\",
        \"Background\": \"Previously led R&D at MegaSoft, with a PhD in Computer Science.\",
        \"School\": \"MIT\",
        \"Area_of_Study\": \"Computer Science\",
        \"Linkedin_URL\": \"https://www.linkedin.com/in/janedoe\"
      }
    ],
    \"Key_Clients\": [
      {
        \"Client\": \"RetailPro Inc.\",
        \"URL\": \"www.retailpro.com\",
        \"Sales_Model\": \"B2B\"
      }
    ],
    \"States\": [
      \"California\",
      \"New York\"
    ],
    \"Office_Locations\": [
      {
        \"State\": \"California\",
        \"City\": \"San Francisco\",
        \"Zip_Code\": \"94105\",
        \"Is_HQ\": true
      },
      {
        \"State\": \"New York\",
        \"City\": \"New York\",
        \"Zip_Code\": \"10001\",
        \"Is_HQ\": false
      }
    ]
  }

  (Real examples should be longer and have all applicable fields and multiple entries as available. Use blanks or \"Information not available\" for missing information.)

  Important:  
  - Follow the correct reasoning-before-conclusion sequence.  
  - Provide only the finalized, correctly structured JSON in your answer."""



  response = client.responses.create(
      model="gpt-5.4",
      input=prompt
  )

  companies.at[i, 'description'] = json.dumps(json.loads(response.output_text))

companies.to_csv("../database/companies.csv", index=False)