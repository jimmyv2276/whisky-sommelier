import openai
import os
from google.colab import userdata
import pandas as pd

OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def generate_explanation(query: str, recommendations_df: pd.DataFrame) -> str:
    """
    Generates a natural language explanation and personalized recommendations using an LLM.
    Weaves the rich database flavor notes directly into a verbose, beautiful sommelier justification
    instead of outputting the raw database review description.

    Args:
        query (str): The user's original search query.
        recommendations_df (pd.DataFrame): DataFrame containing the recommended whiskies.

    Returns:
        str: A natural language explanation and recommendation summary from the LLM.
    """
    if client is None:
        return "Error: OpenAI API client not initialized. Please set your API key."

    # Construct a detailed prompt for the LLM with strict budget guidelines, deep matching, and field requirements
    prompt_parts = [
        "You are an elite, highly knowledgeable whisky sommelier.",
        "Your goal is to explain to a customer exactly WHY each recommended whisky matches their unique query.",
        "",
        "CRITICAL INSTRUCTIONS ON THE MATCH EXPLANATION & FORMAT:",
        "For EACH budget category ('Within Budget', 'Value Option', 'Stretch Option'), you must present the recommended whiskies as a numbered list starting at 1.",
        "",
        "For EACH whisky in that list, use this exact structure:",
        "1. **[Whisky Name]** (The title itself must be a numbered list item)",
        "   - **Rating:** [Show rating points, e.g. 90/100 or 90] (This and subsequent lines MUST be bullet points, NOT numbers)",
        "   - **Price:** $[Show the price]",
        "   - **Why we chose this for you:** [Provide a highly detailed, verbose, and engaging sommelier note. You must explicitly analyze and highlight the core flavor profile, barrel characteristics, and sensory descriptors from the database description (such as fruits, spices, smoke, sugars, etc.) and explain elegantly how they satisfy the user's specific request. Do NOT print the raw database description as a separate field; instead, seamlessly integrate its highlights into this narrative.]",
        "",
        "Do NOT output any separate 'Original Description' field. Do NOT output the numbers 2, 3, 4, or 5 for the attributes. The numbers 1, 2, 3 etc. are ONLY for labeling different whiskies. Attributes under the whisky must ALWAYS be standard bullet points (-).",
        "",
        "ADDITIONAL VALUE & STRETCH GUIDANCE:",
        "- For the 'Value Option', explain why it is a fantastic budget-friendly find that doesn't compromise on their desired profile.",
        "- For the 'Stretch Option', explain the premium complexity they gain if they decide to invest a little more.",
        "",
        "CRITICAL INSTRUCTION ON BUDGETS AND CATEGORIES:",
        "1. Present each whisky strictly under its assigned 'budget_category'. Do NOT reclassify any item.",
        "2. Organise your response clearly by markdown headers: '### Within Budget', '### Value Option', and '### Stretch Option'.",
        "3. Start a new numbered list starting at 1 (e.g., '1.') for each individual budget section.",

        f"\nThe user's original query was: '{query}'.",
        "\nHere are the details of the recommended whiskies retrieved from our database:",
    ]

    for i, row in recommendations_df.iterrows():
        category_label = row.get('budget_category', 'Uncategorized')
        prompt_parts.append(
            f"\nWhisky Name: {row['name']}\n"
            f"Assigned Budget Category: {category_label}\n"
            f"Category: {row['category']}\n"
            f"Rating: {row['rating']}\n"
            f"Price: ${row['price']:.2f}\n"
            f"Database Description/Review: {row['description']}\n"
        )

    prompt_parts.append(
        "\nBased on these details, write your enthusiastic, tailored sommelier response. Use bolding and structured numbered lists for the whiskies under each section, and bullet points for the attributes to ensure that Price, Rating, and 'Why we chose this for you' (with deeply integrated flavor notes) are clearly and cleanly outputted for every single recommendation."
    )

    full_prompt = "\n".join(prompt_parts)

    try:
        response = client.chat.completions.create(
            model="gpt-6.1-sol",  # 1. Updated to Sol model
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly analytical, passionate, and structured whisky sommelier "
                        "who justifies every recommendation by linking sensory review descriptors "
                        "directly back to the customer's query. You always list whiskies as numbered "
                        "list items, and their attributes (Price, Rating, and Why we chose this) "
                        "strictly as bullet points underneath them. You never output a separate "
                        "'Original Description' line; instead, you weave flavor notes directly "
                        "into the 'Why we chose this' explanation."
                    )
                },
                {"role": "user", "content": full_prompt}
            ],
            max_completion_tokens=1500  # 2. Kept max_completion_tokens (required for Sol)
            # temperature=0.7           # 3. REMOVED temperature parameter to prevent 400 error
        )
        return response.choices[0].message.content

    except openai.OpenAIError as e:  # 4. Correct modern SDK error handler
        return f"OpenAI API error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"        
