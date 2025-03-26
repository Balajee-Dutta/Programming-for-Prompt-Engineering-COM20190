# =============================================================================
# MODULE 1: IMPORTS AND SETUP
# =============================================================================
import pandas as pd
from textblob import TextBlob

# =============================================================================
# MODULE 2: DATA LOADING AND CLEANING
# - Reads the Excel file and strips any extra whitespace from column names.
#   Ensures required columns if they exist. Otherwise they default to "Unknown".
# =============================================================================
def load_and_clean_data(file_name: str) -> pd.DataFrame:
    df = pd.read_excel(file_name)
    df.columns = df.columns.str.strip()  # Remove extra whitespace from column names
    return df

# =============================================================================
# MODULE 3: SENTIMENT ANALYSIS PER ENTRY (Using TextBlob)
# - We define five aspects and related keywords.
#   For each aspect, if the keywords are found in the feedback, we perform
#   sentiment analysis on the entire feedback (TextBlob). Otherwise, we label
#   the aspect sentiment as "N/A".
# =============================================================================
def get_aspect_sentiment(user_feedback: str) -> dict:
    """
    Returns a dictionary of sentiments for each aspect based on simple
    keyword matching within the user's feedback text. Uses TextBlob for polarity.
    """

    # Define aspects and their associated keywords, with an expanded vocabulary
    aspects_keywords = {
        "Customer Support & Issue Resolution": [
            "support", "issue", "help", "resolve", "resolution", "complaint",
            "service", "assistance", "contact", "agent", "problem", "refund"
        ],
        "Cancellation & Driver Availability": [
            "cancel", "cancellation", "driver availability", "availability",
            "schedule", "time slot", "driver shortage", "unable to find driver"
        ],
        "Ride Comfort & Vehicle Condition": [
            "comfort", "vehicle", "car", "seat", "clean", "ride comfort",
            "odor", "air conditioning", "music volume", "noise", "space"
        ],
        "Trip Efficiency & Route Accuracy": [
            "efficiency", "route", "gps", "wrong route", "timely", "late",
            "delay", "traffic", "shortcuts", "navigation", "fast", "speed"
        ],
        "Billing Transparency": [
            "bill", "billing", "fare", "payment", "charge", "transparent",
            "hidden fees", "receipt", "tax", "cost breakdown", "price"
        ],
    }

    # Convert feedback to lowercase for easy matching
    feedback_lower = user_feedback.lower()

    def polarity_to_label(polarity: float) -> str:
        # Convert TextBlob polarity to a sentiment label
        if polarity > 0.05:
            return "Positive"
        elif polarity < -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Perform sentiment analysis for relevant aspects
    aspect_sentiments = {}
    for aspect, keywords in aspects_keywords.items():
        # Check if any keyword is present in the feedback for this aspect
        if any(keyword in feedback_lower for keyword in keywords):
            polarity = TextBlob(user_feedback).sentiment.polarity
            aspect_sentiments[aspect] = polarity_to_label(polarity)
        else:
            aspect_sentiments[aspect] = "N/A"

    return aspect_sentiments

def perform_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates over each row, retrieves user feedback and other columns,
    and analyzes sentiment using TextBlob. Returns a new DataFrame with
    the analysis results for each row.
    """
    analysis_results = []

    for idx, row in df.iterrows():
        user_name = row.get("userName", None)
        if pd.isna(user_name) or user_name is None:
            user_name = f"User {idx+1}"  # Fallback if userName is not available

        user_feedback = row.get("User Feedback", None)
        if pd.isna(user_feedback) or user_feedback is None:
            user_feedback = "Unknown"

        location = row.get("Location", None)
        if pd.isna(location) or location is None:
            location = "Unknown"

        driver_name = row.get("Driver Name", "Unknown")

        rating = row.get("Rating", 0)
        if pd.isna(rating):
            rating = 0

        # Get aspect-based sentiment labels
        aspect_sentiments = get_aspect_sentiment(user_feedback)

        # Build a simple string summary of the aspects
        analysis_text = (
            f"User Feedback: {user_feedback}\n\n"
            f"Customer Support- {aspect_sentiments['Customer Support & Issue Resolution']}\n"
            f"Cancellation- {aspect_sentiments['Cancellation & Driver Availability']}\n"
            f"Ride Comfort- {aspect_sentiments['Ride Comfort & Vehicle Condition']}\n"
            f"Trip Efficiency- {aspect_sentiments['Trip Efficiency & Route Accuracy']}\n"
            f"Billing- {aspect_sentiments['Billing Transparency']}"
        )

        # Print analysis to console for each row, referencing the user
        print(f"--- Analysis for {user_name} ---")
        print(f"Driver Name: {driver_name}, Location: {location}")
        print(analysis_text, "\n")

        analysis_results.append({
            "userName": user_name,
            "Driver Name": driver_name,
            "Location": location,
            "User Feedback": user_feedback,
            "Rating": rating,
            "Analysis": analysis_text,
            "Customer Support & Issue Resolution": aspect_sentiments["Customer Support & Issue Resolution"],
            "Cancellation & Driver Availability": aspect_sentiments["Cancellation & Driver Availability"],
            "Ride Comfort & Vehicle Condition": aspect_sentiments["Ride Comfort & Vehicle Condition"],
            "Trip Efficiency & Route Accuracy": aspect_sentiments["Trip Efficiency & Route Accuracy"],
            "Billing Transparency": aspect_sentiments["Billing Transparency"]
        })

    return pd.DataFrame(analysis_results)

# =============================================================================
# MODULE 4: AGGREGATION BY DRIVER
# - Groups feedback and analysis, calculates the average rating for each driver.
# =============================================================================
def aggregate_analysis_by_driver(analysis_df: pd.DataFrame) -> pd.DataFrame:
    driver_summary = (
        analysis_df
        .groupby("Driver Name")
        .agg({
            "User Feedback": lambda x: " ||| ".join(x),
            "Analysis": lambda x: " ||| ".join(x),
            "Rating": "mean"
        })
        .reset_index()
    )

    # Print the aggregated data
    print("=== DRIVER SUMMARY (Aggregated) ===")
    print(driver_summary, "\n")
    return driver_summary

# =============================================================================
# MODULE 5: FINAL SUMMARIZATION
# - Summarizes each driver's performance based on average rating. If avg rating
#   < 3, it only highlights negative aspects. Otherwise, it highlights the
#   positive ones.
# =============================================================================
def generate_summary_for_drivers(driver_summary: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    final_summaries = []

    # List of aspect columns in the analysis_df
    aspect_cols = [
        "Customer Support & Issue Resolution",
        "Cancellation & Driver Availability",
        "Ride Comfort & Vehicle Condition",
        "Trip Efficiency & Route Accuracy",
        "Billing Transparency"
    ]

    def count_aspect_sentiment(df_for_driver: pd.DataFrame, aspect: str, sentiment_target: str) -> int:
        """
        Counts how many times a given sentiment_target (e.g., "Negative", "Positive")
        appears for a particular aspect among the driver's rows.
        """
        return len(df_for_driver[df_for_driver[aspect] == sentiment_target])

    for _, row in driver_summary.iterrows():
        driver_name = row["Driver Name"]
        avg_rating = row["Rating"]

        # Filter the main analysis_df for this driver
        driver_data = analysis_df[analysis_df["Driver Name"] == driver_name]

        if avg_rating < 3:
            # Focus on negative aspects
            aspect_counts = {
                aspect: count_aspect_sentiment(driver_data, aspect, "Negative")
                for aspect in aspect_cols
            }
            # Identify the most frequently negative aspect
            worst_aspect = max(aspect_counts, key=aspect_counts.get)
            # If there's zero negative mentions for all aspects, handle gracefully
            if aspect_counts[worst_aspect] == 0:
                worst_aspect = "No negative aspect found"

            summary_text = (
                f"Driver {driver_name} is performing poorly (Average Rating: {avg_rating:.2f}).\n"
                f"One of the most frequent negative aspects is: {worst_aspect}.\n"
                f"Suggestion: Improve on {worst_aspect} to enhance rider satisfaction."
            )
        else:
            # Focus on positive aspects
            aspect_counts = {
                aspect: count_aspect_sentiment(driver_data, aspect, "Positive")
                for aspect in aspect_cols
            }
            # Identify the most frequently positive aspect
            best_aspect = max(aspect_counts, key=aspect_counts.get)
            # If there's zero positive mentions for all aspects, handle gracefully
            if aspect_counts[best_aspect] == 0:
                best_aspect = "No positive aspect found"

            summary_text = (
                f"Driver {driver_name} is performing well (Average Rating: {avg_rating:.2f}).\n"
                f"One of the most frequent positive aspects is: {best_aspect}.\n"
                f"Suggestion: Continue to maintain strengths in {best_aspect}!"
            )

        # Print final summary for this driver
        print(f"=== Final Summary for Driver: {driver_name} ===")
        print(summary_text, "\n")

        final_summaries.append({
            "Driver Name": driver_name,
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
    # 1) Load and clean data; replace the file name with your actual file if needed
    df = load_and_clean_data("uber user feedback-DATAset(Driver Details).xlsx")
    
    # 2) Perform sentiment analysis
    analysis_df = perform_sentiment_analysis(df)
    print("=== ANALYSIS RESULTS DATAFRAME ===")
    print(analysis_df, "\n")

    # 3) Aggregate by driver
    driver_summary = aggregate_analysis_by_driver(analysis_df)

    # 4) Generate final summaries
    final_summaries = generate_summary_for_drivers(driver_summary, analysis_df)
    print("=== FINAL SUMMARIES DATAFRAME ===")
    print(final_summaries, "\n")

    # (Optional) Save final summaries to an Excel file
    # final_summaries.to_excel("driver_performance_summaries.xlsx", index=False)

if __name__ == "__main__":
    main()
