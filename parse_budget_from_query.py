import json
import pandas as pd
import numpy as np

def parse_budget(query: str):
    """
    Extracts price boundaries from the user query.
    Supports patterns like '$75 or less', 'under 100', '$50-$100', 'between 50 and 100'
    """
    query_clean = query.lower().replace('$', '')

    # Try to find ranges: "50 to 100", "50-100", "between 50 and 100"
    range_match = re.search(r'(\d+)\s*(?:to|-|and)\s*(\d+)', query_clean)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    # Try to find upper bounds: "less than 75", "under 75", "75 or less", "max 75"
    upper_match = re.search(r'(?:under|less than|or less|max|budget of|up to)\s*(\d+)|(\d+)\s*(?:or less|max)', query_clean)
    if upper_match:
        val = upper_match.group(1) or upper_match.group(2)
        return 0.0, float(val)

    # Try to find lower bounds: "above 50", "more than 50", "at least 50"
    lower_match = re.search(r'(?:above|more than|at least|over|min)\s*(\d+)', query_clean)
    if lower_match:
        val = lower_match.group(1)
        return float(val), float('inf')

    # Fallback: check if a single price is mentioned as max price
    single_match = re.search(r'\b(\d+)\b', query_clean)
    if single_match:
        return 0.0, float(single_match.group(1))

    return None, None

def generate_search_plan(user_query: str) -> dict:
    """
    Uses the LLM as a Query Planner to extract structured filters and
    rewrite the user's search query into clean, objective flavor descriptors.
    """
    system_instruction = (
        "You are an expert search planner for a whisky database. Your task is to analyze the user's query "
        "and extract search filters, exclusions, and clean semantic descriptors in a strict JSON format."
    )

    prompt = f"""
    Analyze the following user query: "{user_query}"

    Generate a JSON response with these exact keys:
    1. "semantic_search_query": Translate brand preferences or high-level styles into precise, sensory flavor descriptors (e.g., if they like Oban but want something else, translate Oban's style to descriptors like 'salted caramel, orange peel, light smoke, coastal air, heather'). Do not include the name of any excluded brands or negated descriptors in this search string.
    2. "price_range": An object with "min" (float/null) and "max" (float/null) prices parsed from the query.
    3. "exclude_brands": An array of strings representing brand names or distillery names the user explicitly wants to avoid (e.g. ['Oban']).
    4. "exclude_keywords": An array of sensory terms or characteristics the user explicitly wants to avoid (e.g. ['medicinal', 'iodine', 'band-aid', 'peat']).

    Query: "{user_query}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-6.1-sol",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
            # temperature=0.0 <-- REMOVED this parameter to allow the default (1.0)
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"Warning: Failed to generate search plan due to {e}. Falling back to default values.")
        return {
            "semantic_search_query": user_query,
            "price_range": {"min": None, "max": None},  # Fixed 'null' to Python's 'None'
            "exclude_brands": [],
            "exclude_keywords": []
        }

def search_whiskies_with_planner(user_query: str, target_k: int = 5) -> pd.DataFrame:
    """
    Performs an LLM-guided search that dynamically applies brand exclusions, flavor exclusions,
    and customized price filters based on the generated search plan.
    """
    # 1. Generate the structured search plan
    plan = generate_search_plan(user_query)
    semantic_query = plan.get("semantic_search_query", user_query)
    price_info = plan.get("price_range", {}) or {}
    min_p = price_info.get("min")
    max_p = price_info.get("max")
    exclude_brands = plan.get("exclude_brands", []) or []
    exclude_keywords = plan.get("exclude_keywords", []) or []

    print(f"\n[Planner] Rewritten Semantic Query: \"{semantic_query}\"")
    print(f"[Planner] Price Filters: Min = {min_p}, Max = {max_p}")
    print(f"[Planner] Exclude Brands: {exclude_brands} | Exclude Keywords: {exclude_keywords}")

    # 2. Get embeddings based strictly on the semantic, clean flavor representation
    query_embedding = model.encode([semantic_query])
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Fetch a wider pool (150 candidates) to allow room for exclusion filtering
    distances, indices = active_faiss_index.search(query_embedding, 150)
    candidates = df_cleaned.iloc[indices[0]].copy()

    # 3. Apply Exclusions dynamically
    for brand in exclude_brands:
        candidates = candidates[~candidates['name'].str.lower().str.contains(brand.lower())]

    for keyword in exclude_keywords:
        candidates = candidates[~candidates['description'].str.lower().str.contains(keyword.lower())]

    # 4. Handle default/derived budgets
    if min_p is None: min_p = 0.0
    if max_p is None: max_p = float('inf')

    val_threshold = min_p if min_p > 0 else (max_p * 0.5 if max_p != float('inf') else 50.0)

    # 5. Distribute candidates into strict budget pools
    within_budget = candidates[(candidates['price'] >= min_p) & (candidates['price'] <= max_p)].copy()
    within_budget['budget_category'] = 'Within Budget'

    value_options = candidates[candidates['price'] < val_threshold].copy()
    value_options['budget_category'] = 'Value Option'

    stretch_min = max_p if max_p != float('inf') else 150.0
    stretch_max = stretch_min * 1.35
    stretch_options = candidates[(candidates['price'] > stretch_min) & (candidates['price'] <= stretch_max)].copy()
    stretch_options['budget_category'] = 'Stretch Option'

    selected_blocks = []

    # Gather distributed recommendations
    selected_blocks.append(within_budget.head(3))

    if not value_options.empty:
        selected_blocks.append(value_options.head(1))
    else:
        cheapest_fallback = candidates.nsmallest(1, 'price').copy()
        cheapest_fallback['budget_category'] = 'Value Option'
        selected_blocks.append(cheapest_fallback)

    if not stretch_options.empty:
        selected_blocks.append(stretch_options.head(1))

    # Deduplicate and combine
    final_df = pd.concat(selected_blocks).drop_duplicates(subset=['name'])

    # Backfill if we are under the target count
    if len(final_df) < target_k:
        fill_count = target_k - len(final_df)
        remaining = candidates[~candidates['name'].isin(final_df['name'])].head(fill_count).copy()
        remaining['budget_category'] = 'Alternative Option'
        final_df = pd.concat([final_df, remaining])

    return final_df.head(target_k)

print("Planner-based search pipeline defined successfully!")