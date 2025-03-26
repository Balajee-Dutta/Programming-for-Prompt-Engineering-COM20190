# Programming-for-Prompt-Engineering-COM20190

# Aspect-Based Sentiment Analysis of Uber User Reviews

This repository focuses on identifying user sentiment for key aspects of Uber’s ride experience (e.g., Customer Support, Cancellation, Ride Comfort, etc.) from real user feedback data.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Dataset](#dataset)  
4. [Python Scripts](#python-scripts)  
5. [Output Files](#output-files)  
6. [How to Use](#how-to-use)  
7. [Potential Improvements](#potential-improvements)  
8. [Contact](#contact)

---

## Project Overview

This project applies **Aspect-Based Sentiment Analysis** techniques to categorize user feedback into distinct aspects (e.g., Customer Support, Cancellation, Billing) and assign sentiments (Positive, Negative, Neutral, or N/A). Two primary approaches have been implemented:

1. **OpenAI Model-Based Approach**: Uses GPT-like models (as demonstrated in the `Prompt-*` scripts) to automatically label each feedback segment.
2. **TextBlob–Based Approach**: Leverages the TextBlob library’s built-in sentiment analysis and keyword matching to process feedback.

---

## Repository Structure

The repository contains the following core files and directories:

|API Programming- output- file.docx| Prompt- Ub-Sn-4o mini model.py |Prompt-Ub-Sn-4o model.py |Text Blob- file.py |Text Blob- output -file.docx| uber user feedback-DATAset(Driver Details).xlsx| README.md <-- (This File)


| File / Folder                               | Description                                                                                                                                   |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **API Programming- output- file.docx**      | Shows all (60 + 60 user Feedback) output from a GPT-based script where an external API call was used for aspect extraction and sentiment classification.                 |
| **Prompt- Ub-Sn-4o mini model.py**           | Demonstrates a Python script calling a GPT-like model (in “mini” form) to perform aspect-based sentiment analysis on the Uber feedback dataset. |
| **Prompt-Ub-Sn-4o model.py**                 | Similar to the “mini model” but relies on a different GPT-based model. Handles data loading, sentiment classification, and result aggregation. |
| **Text Blob- file.py**                       | Demonstrates use of the TextBlob library for aspect-based sentiment analysis. Scans user feedback for keywords and applies sentiment analysis.  |
| **Text Blob- outfut -file.docx**             | Shows the final labeled output generated by the `Text Blob- file.py` script, highlighting aspect-based sentiment results.                       |
| **uber user feedback-DATAset(Driver Details).xlsx** | Contains the dataset of Uber user reviews. Each row includes user feedback, driver name, location, and ratings. The Python scripts read and analyze this file. |

---

## Dataset

- **File Name**: `uber user feedback-DATAset(Driver Details).xlsx`
- **Contents**:
  - **Driver Name** – The name of the Uber driver mentioned in the feedback.
  - **Location** – The city or region where the ride took place.
  - **User Feedback** – Written feedback from the user.
  - **Rating** – Numeric or categorical rating if available.

This file is crucial for running the scripts. It should be placed in the same directory or the file path must be appropriately updated within the scripts.

---

## Python Scripts

### 1. Prompt- Ub-Sn-4o mini model.py
- **Purpose**: Performs aspect-based sentiment analysis by sending user feedback to a GPT-like model (“GPT-4o-mini”) for classification.
- **Key Sections**:
  - **Data Loading & Cleaning**: Reads the Excel data and handles missing or NaN fields.
  - **Sentiment Analysis**: Constructs a prompt for each feedback entry, calls the GPT-based model, and outputs aspect-by-aspect sentiment.
  - **Aggregation & Summaries**: Groups results by driver and calculates average ratings, followed by final recommendations.

### 2. Prompt-Ub-Sn-4o model.py
- **Purpose**: Similar to the “mini” version but uses “gpt-4o.” Processes data by performing aspect-based sentiment analysis, aggregation, and final summarization.

### 3. Text Blob- file.py
- **Purpose**: Employs the **TextBlob** library for aspect-based sentiment analysis. Keyword checks are performed on user feedback to determine whether each aspect is relevant, and TextBlob is used to assign a sentiment label.
- **Process**:
  1. **Data Loading**: Reads from the Excel file.
  2. **Keyword Matching**: Identifies relevant words (e.g., “support,” “cancel,” “bill”) for each aspect.
  3. **Sentiment Polarity**: TextBlob calculates polarity on the entire feedback. The polarity is mapped to Positive, Negative, or Neutral.
  4. **Output**: Prints a summary indicating each aspect and the corresponding sentiment.

---

## Output Files

### 1. API Programming- output- file.docx
Includes a sample of how the GPT-based script categorized aspects and assigned sentiments within user feedback.

### 2. Text Blob- outfut -file.docx
Demonstrates the final labeled output produced by `Text Blob- file.py`, displaying how each review is categorized into aspects with respective sentiment labels.

These files can help verify and compare results from different sentiment-analysis approaches.

---

## How to Use

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Balajee-Dutta/Programming-for-Prompt-Engineering-COM20190.git
   cd Programming-for-Prompt-Engineering-COM20190

Install Dependencies
Python 3+ is required. Install the necessary libraries:
pip install pandas openpyxl textblob

For GPT-based scripts, an API key for OpenAI or the relevant GPT provider is required. Insert the key in the indicated lines (openai.api_key = ["insert here"]).

Run the Scripts

GPT-based Approach:
python Prompt- Ub-Sn-4o mini model.py

and

python Prompt-Ub-Sn-4o model.py

Ensure that Uber user feedback-DATAset(Driver Details).xlsx is located in the same folder or update the file path in the script if necessary.

TextBlob-Based Approach:
python Text Blob- file.py

The console output will display sentiment results for each user review.

Review the Output
The scripts print sentiment analysis for each review. Additionally, the corresponding .docx files (API Programming- output- file.docx and Text Blob- output -file.docx) contain sample results.

\\\\//////
Contact
Questions, suggestions, or contributions may be directed via GitHub Issues or email:

Author: [Balajee-Dutta]

Project: Aspect-Based Sentiment Analysis for Uber User Reviews



