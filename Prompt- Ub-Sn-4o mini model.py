# =============================================================================
# MODULE 1: IMPORTS AND SETUP
# - This section imports the required libraries and sets the OpenAI API key.
# =============================================================================
import openai
import pandas as pd

# Replace "your-api-key-here" with your actual API key
openai.api_key = ["insert here"]

# =============================================================================
# MODULE 2: DATA LOADING AND CLEANING
# - Reads the Excel file and strips any extra whitespace from column names.
#   Ensures "User Feedback" and "Location" default to "Unknown" if missing.
# =============================================================================
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    return df

# =============================================================================
# MODULE 3: SENTIMENT ANALYSIS PER ENTRY
# - Iterates over each feedback entry, calls the OpenAI model, and returns
#   a list of analysis results. Defaults to "Unknown" for missing user feedback
#   or location. Prints each analysis result.
# =============================================================================
def perform_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    analysis_results = []
    for idx, row in df.iterrows():
        # Fetch userName if available, otherwise default to "User <idx+1>"
        user_name = row.get('userName', None)
        if pd.isna(user_name) or user_name is None:
            user_name = f"User {idx + 1}"

        user_feedback = row.get('User Feedback', None)
        if pd.isna(user_feedback) or user_feedback is None:
            user_feedback = "Unknown"
        
        location = row.get('Location', None)
        if pd.isna(location) or location is None:
            location = "Unknown"
        
        driver_name = row.get('Driver Name', "Unknown")
        rating = row.get('Rating', 0)
        if pd.isna(rating):
            rating = 0

        # Updated aspects_keywords with more relatable words
        prompt = f"""
You are an expert in sentiment analysis for customer reviews.

We have the following aspects (including synonyms to broaden scope):
1) Customer Support, Issue Resolution, Communication Quality
2) Cancellation, Driver Availability, Pickup Timeliness
3) Ride Comfort, Vehicle Condition, Cleanliness
4) Trip Efficiency, Route Accuracy, Journey Duration
5) Billing Transparency, Fare Clarity, Payment Process

We have 4 possible sentiments: Positive, Negative, Neutral, N/A

User Feedback: {user_feedback}
Driver Name: {driver_name}
Location: {location}
Rating: {rating}

Please return your response in this exact format:

User Feedback: {user_feedback}

Customer Support- <sentiment>
Cancellation- <sentiment>
Ride Comfort- <sentiment>
Trip Efficiency- <sentiment>
Billing- <sentiment>
"""
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        analysis_text = response["choices"][0]["message"]["content"].strip()

        # Print each row's analysis, now mentioning the userName or User <idx+1>
        print(f"--- Analysis for {user_name} ---")
        print(f"Driver Name: {driver_name}, Location: {location}")
        print(analysis_text, "\n")

        analysis_results.append({
            "Driver Name": driver_name,
            "Location": location,
            "User Feedback": user_feedback,
            "Rating": rating,
            "Analysis": analysis_text
        })

    return pd.DataFrame(analysis_results)

# =============================================================================
# MODULE 4: AGGREGATION BY DRIVER
# - Groups feedback, analysis, and calculates the average rating for each driver.
# =============================================================================
def aggregate_analysis_by_driver(analysis_df: pd.DataFrame) -> pd.DataFrame:
    driver_summary = analysis_df.groupby("Driver Name").agg({
        "User Feedback": lambda x: " ||| ".join(x),
        "Analysis": lambda x: " ||| ".join(x),
        "Rating": "mean"
    }).reset_index()

    # Print the aggregated data
    print("=== DRIVER SUMMARY (Aggregated) ===")
    print(driver_summary, "\n")
    return driver_summary

# =============================================================================
# MODULE 5: FINAL SUMMARIZATION
# - Creates a summary for each driver, referencing only negative aspects if
#   average rating < 3, otherwise referencing only positive aspects.
#   Prints each driver's final summary, now also including the average rating.
# =============================================================================
def generate_summary_for_drivers(driver_summary: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    final_summaries = []
    for _, row in driver_summary.iterrows():
        driver_name = row["Driver Name"]
        feedback = row["User Feedback"]
        analysis = row["Analysis"]
        avg_rating = row["Rating"]
        
        matching = original_df[original_df["Driver Name"] == driver_name]
        if not matching.empty and "Location" in matching.columns:
            unique_locations = matching["Location"].unique()
            loc_str = ", ".join(str(loc) for loc in unique_locations if not pd.isna(loc))
            if not loc_str:
                loc_str = "Unknown"
        else:
            loc_str = "Unknown"

        prompt = f"""
As a dedicated researcher in sentiment analysis with a focus on evaluating driver performance, examine the following data:

Driver Name: {driver_name}
Location(s): {loc_str}
Average Rating: {avg_rating:.2f}

Aggregated User Feedback:
{feedback}

Aggregated Sentiment Analysis:
{analysis}

Only mention negative aspects if the average rating < 3, and only mention positive aspects if the average rating >= 3.
If average rating >= 3, the driver is considered "good" overall; otherwise "poor."

Output- 
Driver {driver_name}, constantly performing <good/poor>, one of the repetitive callouts is <reason>.
Suggestion: <improvement suggestion>.
"""
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        summary_text = resp["choices"][0]["message"]["content"].strip()

        # Print each driver's final summary with the updated format
        print(f"=== Final Summary for Driver: {driver_name} ===")
        print(f"(Average Rating: {avg_rating:.2f})\n")
        print(summary_text, "\n")

        final_summaries.append({
            "Driver Name": driver_name,
            "Location(s)": loc_str,
            "Average Rating": avg_rating,
            "Summary": summary_text
        })
    return pd.DataFrame(final_summaries)

# =============================================================================
# MODULE 6: MAIN FUNCTION
# - Orchestrates the flow: data loading, sentiment analysis, aggregation,
#   and final summarization. Prints the final summaries DataFrame at the end.
# =============================================================================
def main():
    df = load_and_clean_data("uber user feedback-DATAset(Driver Details).xlsx")
    analysis_df = perform_sentiment_analysis(df)
    print("=== ANALYSIS RESULTS DATAFRAME ===")
    print(analysis_df, "\n")

    driver_summary = aggregate_analysis_by_driver(analysis_df)
    final_summaries = generate_summary_for_drivers(driver_summary, analysis_df)

    print("=== FINAL SUMMARIES DATAFRAME ===")
    print(final_summaries, "\n")


if __name__ == "__main__":
    main()
