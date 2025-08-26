import os
import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from playwright.sync_api import sync_playwright
from textblob import TextBlob


# Initialize Flask App
app = Flask(__name__, static_folder="static")
CORS(app)  

# Load Models
print("🔄 Loading trained models...")
churn_rf_model = joblib.load("models/churn_rf.pkl")  # Random Forest Churn Model
churn_xgb_model = joblib.load("models/churn_xgb.pkl")  # XGBoost Churn Model
segmentation_model = joblib.load("models/customer_segmentation.pkl")
recommendation_model = joblib.load("models/product_svd.pkl")
print("✅ Models loaded successfully!")

# Load Processed Data
df = pd.read_csv("data/processed_customer_data.csv")

# ---- Enable CORS for Specific Routes ----
@app.after_request
def add_cors_headers(response):
    """Adds CORS headers to allow frontend requests from different ports."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# ---- API: Serve EDA Charts ----
@app.route("/api/charts/<chart_name>", methods=["GET"])
def serve_chart(chart_name):
    chart_dir = "static/charts"
    valid_charts = {
        "spending_distribution": "spending_distribution.png",
        "churn_vs_spending": "churn_vs_spending.png",
        "customer_age_distribution": "customer_age_distribution.png",
        "feature_correlation_heatmap": "feature_correlation_heatmap.png",
        "churn_confusion_matrix_rf": "churn_confusion_matrix_rf.png",
        "churn_confusion_matrix_xgb": "churn_confusion_matrix_xgb.png",
        "churn_feature_importance": "churn_feature_importance.png",
        "customer_elbow_method": "customer_elbow_method.png",
        "sentiment_distribution": "sentiment_distribution.png"
    }


    if chart_name not in valid_charts:
        return jsonify({"error": "Invalid chart name"}), 400

    return send_from_directory(chart_dir, valid_charts[chart_name])

# ---- API: Dashboard Metrics ----
@app.route("/api/dashboard-metrics", methods=["GET"])
def dashboard_metrics():
    """Provides Total Purchases, Active Users, and Feedback Score for Dashboard"""
    total_purchases = int(df["Total Purchase Amount"].sum())
    active_users = df["Customer ID"].nunique()
    feedback_score = round(np.random.uniform(3.5, 5.0), 1)  # Placeholder Score

    return jsonify({
        "total_purchases": total_purchases,
        "active_users": active_users,
        "feedback_score": feedback_score
    })

# ---- API: Search ----
import traceback
@app.route("/api/search", methods=["GET"])
def search():
    """Search for customers by name or product category across both datasets"""

    query = request.args.get("query", "").strip().lower()

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        # Load both datasets
        df1 = pd.read_csv("data/ecommerce_customer_data_custom_ratios.csv")
        df2 = pd.read_csv("data/ecommerce_customer_data_large.csv")

        # Combine both datasets
        df = pd.concat([df1, df2], ignore_index=True)

        # Ensure all product categories are included
        all_products = df["Product Category"].dropna().unique().tolist()

        # Search for customer name or product category
        results = df[
            df["Customer Name"].str.lower().str.contains(query, na=False) |
            df["Product Category"].str.lower().str.contains(query, na=False)
        ]

        response_data = results[["Customer ID", "Customer Name", "Product Category", "Total Purchase Amount"]].head(10).to_dict(orient="records")

        return jsonify({
            "search_results": response_data,
            "all_products": all_products  
        })

    except Exception as e:
        print("Error:", str(e))  
        traceback.print_exc()  
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


# ---- API: Purchase Trends (Chart.js) ----
@app.route("/api/purchase-trends", methods=["GET"])
def purchase_trends():
    """Returns Monthly Purchase Data for Chart Visualization"""
    df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")
    monthly_sales = df.groupby(df["Purchase Date"].dt.to_period("M"))["Total Purchase Amount"].sum()

    return jsonify({
        "labels": monthly_sales.index.astype(str).tolist(),
        "values": monthly_sales.values.tolist()
    })

# ---- API: Customer Segmentation (D3.js) ----
@app.route("/api/customer-segmentation", methods=["GET"])
def customer_segmentation():
    """Returns Customer Segments and Counts"""
    segment_counts = df["Segment"].value_counts().reset_index()
    segment_counts.columns = ["label", "value"]

    return jsonify(segment_counts.to_dict(orient="records"))

# ---- API: Churn Prediction ----
@app.route("/predict_churn", methods=["POST"])
def predict_churn():
    data = request.get_json()
    churn_features = ["Total Spending", "Purchase Frequency", "Return Rate"]
    model_type = data.get("model", "rf")  # Choose model (default: "rf" for Random Forest)

    try:
        X_input = pd.DataFrame([data])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_input[churn_features])

        if model_type == "xgb":
            churn_prediction = churn_xgb_model.predict(X_scaled)
            churn_prob = churn_xgb_model.predict_proba(X_scaled)[:, 1]
        else:
            churn_prediction = churn_rf_model.predict(X_scaled)
            churn_prob = churn_rf_model.predict_proba(X_scaled)[:, 1]

        return jsonify({
            "churn_prediction": int(churn_prediction[0]),
            "churn_probability": float(churn_prob[0]),
            "model_used": "XGBoost" if model_type == "xgb" else "Random Forest"
        })
    except Exception as e:
        return jsonify({"error": f"Churn prediction failed: {str(e)}"}), 400

# ---- API: Customer Segmentation ----
@app.route("/predict_segment", methods=["POST"])
def predict_segment():
    data = request.get_json()
    segmentation_features = ["Recency", "Frequency", "Monetary"]

    try:
        X_input = pd.DataFrame([data])
        segment = segmentation_model.predict(X_input[segmentation_features])

        return jsonify({"predicted_segment": int(segment[0])})
    except Exception as e:
        return jsonify({"error": f"Segmentation prediction failed: {str(e)}"}), 400

# ---- API: Product Recommendation ----
@app.route("/recommend_products", methods=["POST"])
def recommend_products():
    data = request.get_json()
    customer_id = int(data.get("customer_id"))

    if customer_id not in df["Customer ID"].values:
        return jsonify({"error": "Customer ID not found!"})

    try:
        user_item_matrix = df.pivot_table(index="Customer ID", columns="Product Category", values="Total Purchase Amount", aggfunc="sum").fillna(0)
        user_item_sparse = csr_matrix(user_item_matrix.values)

        customer_index = user_item_matrix.index.get_loc(customer_id)
        similarities = cosine_similarity(user_item_sparse[customer_index], user_item_sparse).flatten()
        top_similar_customers = np.argsort(similarities)[::-1][1:6]

        recommended_products = set()
        for similar_customer in top_similar_customers:
            similar_cust_id = user_item_matrix.index[similar_customer]
            top_products = df[df["Customer ID"] == similar_cust_id]["Product Category"].unique()
            recommended_products.update(top_products)

        return jsonify({"recommended_products": list(recommended_products)})
    except Exception as e:
        return jsonify({"error": f"Product recommendation failed: {str(e)}"}), 400

# ---- API: Product fetching through url ----
@app.route("/api/fetch_product_details", methods=["POST"])
def fetch_product_details():
    try:
        data = request.get_json()
        url = data.get("product_url")

        if not url:
            return jsonify({"error": "Product URL is required."}), 400

        if "amazon" in url:
            product_data = scrape_amazon(url)
        elif "flipkart" in url:
            product_data = scrape_flipkart(url)
        else:
            return jsonify({"error": "Only Amazon and Flipkart URLs are supported."}), 400

        return jsonify({
            "product_info": product_data["product_info"],
            "reviews": product_data["reviews"]
        })

    except Exception as e:
        print("Fetch product details error:", e)
        return jsonify({"error": "Failed to fetch product details."}), 500


def compute_sentiment_score(reviews):
    sentiments = []
    for review in reviews:
        blob = TextBlob(str(review))
        sentiments.append(blob.sentiment.polarity)  

    if sentiments:
        avg_polarity = sum(sentiments) / len(sentiments)
        sentiment_score = round(((avg_polarity + 1) / 2) * 5, 2)  
        return sentiment_score
    else:
        return None


# ---- New: Live Sales Summary API ----
@app.route("/api/sales-summary", methods=["POST"])
def sales_summary():
    try:
        payload = request.get_json(force=True) or {}
        category = (payload.get("category") or "").strip()
        product_title = (payload.get("product_title") or "").strip()

        # Ensure purchase date is parsed
        if not np.issubdtype(df["Purchase Date"].dtype, np.datetime64):
            local_df = df.copy()
            local_df["Purchase Date"] = pd.to_datetime(local_df["Purchase Date"], errors="coerce")
        else:
            local_df = df

        # Filter by category or fuzzy match on title tokens (very light heuristic)
        filtered_df = local_df.copy()
        if category:
            filtered_df = filtered_df[filtered_df["Product Category"].astype(str).str.lower().str.contains(category.lower(), na=False)]
        elif product_title:
            tokens = [t for t in product_title.lower().split() if len(t) > 2]
            if tokens:
                mask = False
                for t in tokens[:3]:
                    mask = mask | filtered_df["Product Category"].astype(str).str.lower().str.contains(t, na=False)
                filtered_df = filtered_df[mask]

        if filtered_df.empty:
            filtered_df = local_df

        # Compute totals
        total_sales = float(filtered_df["Total Purchase Amount"].sum()) if "Total Purchase Amount" in filtered_df.columns else 0.0
        total_orders = int(filtered_df.shape[0])

        # Time-based summaries (last 7 and 30 days)
        now = pd.Timestamp.utcnow()
        last_7 = filtered_df[filtered_df["Purchase Date"] >= now - pd.Timedelta(days=7)]
        last_30 = filtered_df[filtered_df["Purchase Date"] >= now - pd.Timedelta(days=30)]

        summary = {
            "total_sales": round(total_sales, 2),
            "total_orders": total_orders,
            "last_7d_sales": round(float(last_7["Total Purchase Amount"].sum()), 2) if not last_7.empty else 0.0,
            "last_30d_sales": round(float(last_30["Total Purchase Amount"].sum()), 2) if not last_30.empty else 0.0,
        }
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": f"Failed to compute sales summary: {str(e)}"}), 500


# ---- New: AI Assist Chat Endpoint ----
try:
    from openai import OpenAI
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    _openai_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
except Exception:
    _openai_client = None


def _build_context_summary(product_info: dict, reviews: list, sales: dict) -> str:
    if not product_info:
        return "No product context provided. Ask the user for a product link (Amazon/Flipkart)."

    title = product_info.get("title") or "Unknown Title"
    price = product_info.get("price") or "N/A"
    rating = product_info.get("rating") or "N/A"
    source = product_info.get("source") or "Unknown"
    sentiment_score = product_info.get("sentiment_score")
    sentiment_str = f"{sentiment_score}/5" if sentiment_score is not None else "N/A"

    sales_bits = []
    if sales:
        if sales.get("last_7d_sales") is not None:
            sales_bits.append(f"7d sales: {sales.get('last_7d_sales')}")
        if sales.get("last_30d_sales") is not None:
            sales_bits.append(f"30d sales: {sales.get('last_30d_sales')}")
    sales_text = ", ".join(sales_bits) if sales_bits else "No recent sales data"

    reviews_text = "; ".join((reviews or [])[:3]) if reviews else "No reviews available"

    return (
        f"Product: {title} (Source: {source})\n"
        f"Price: {price}, Rating: {rating}, Sentiment: {sentiment_str}\n"
        f"Sales: {sales_text}\n"
        f"Sample reviews: {reviews_text}"
    )


def _heuristic_ai_reply(user_message: str, product_info: dict, reviews: list, sales: dict) -> str:
    summary = _build_context_summary(product_info, reviews, sales)
    # Very light rule-based guidance
    pros = []
    cons = []

    sentiment = (product_info or {}).get("sentiment_score")
    if sentiment is not None:
        if sentiment >= 4.0:
            pros.append("Strong positive sentiment from recent reviews.")
        elif sentiment >= 3.0:
            pros.append("Generally favorable sentiment, with some mixed feedback.")
        else:
            cons.append("Below-average sentiment; consider alternatives or check detailed reviews.")

    # Price heuristic if numeric can be parsed
    price_raw = (product_info or {}).get("price") or ""
    price_num = None
    try:
        price_num = float(''.join(ch for ch in price_raw if ch.isdigit() or ch == '.'))
    except Exception:
        pass

    if price_num is not None and price_num > 0:
        if price_num > 50000:  # assume INR high-ticket
            cons.append("High price point; verify that it meets your exact requirements.")
        elif price_num < 2000:
            pros.append("Budget-friendly option.")

    # Sales heuristic
    if sales:
        last_30 = sales.get("last_30d_sales") or 0
        if last_30 > 0:
            pros.append("Recent sales activity suggests market interest.")

    advice = []
    if pros:
        advice.append("Why it could be good: " + " ".join(pros))
    if cons:
        advice.append("Cautions: " + " ".join(cons))

    base = (
        f"Here's a quick summary based on available data:\n\n{summary}\n\n" +
        ("\n".join(advice) if advice else "") +
        ("\n\nTell me your use-case and constraints (budget, size, brand, features) so I can advise if this fits or propose alternatives." )
    )

    if user_message and user_message.strip():
        base += f"\n\nRegarding your note: '{user_message.strip()}', here are tailored thoughts: "
        if "gaming" in user_message.lower():
            base += "Consider performance metrics (CPU/GPU), thermals, and refresh rate."
        elif "battery" in user_message.lower():
            base += "Prioritize battery capacity (mAh/Wh) and real-world reviews on endurance."
        elif "camera" in user_message.lower():
            base += "Check sensor size, OIS, low-light performance, and user photo samples."
        else:
            base += "I can refine suggestions as you share more specifics."

    return base


@app.route("/api/ai/assist", methods=["POST"])
def ai_assist():
    try:
        payload = request.get_json(force=True) or {}
        message = (payload.get("message") or "").strip()
        history = payload.get("history") or []
        product_info = payload.get("product_info") or {}
        reviews = payload.get("reviews") or []

        # gather sales summary for context
        category = product_info.get("category") or product_info.get("source") or ""
        # Compute quick sales summary inline (avoid nested request cycles)
        try:
            sales = None
            if product_info or category:
                if not np.issubdtype(df["Purchase Date"].dtype, np.datetime64):
                    local_df = df.copy()
                    local_df["Purchase Date"] = pd.to_datetime(local_df["Purchase Date"], errors="coerce")
                else:
                    local_df = df
                filtered_df = local_df
                if category:
                    filtered_df = filtered_df[filtered_df["Product Category"].astype(str).str.lower().str.contains(str(category).lower(), na=False)]
                if filtered_df.empty:
                    filtered_df = local_df
                now = pd.Timestamp.utcnow()
                last_7 = filtered_df[filtered_df["Purchase Date"] >= now - pd.Timedelta(days=7)]
                last_30 = filtered_df[filtered_df["Purchase Date"] >= now - pd.Timedelta(days=30)]
                sales = {
                    "last_7d_sales": round(float(last_7["Total Purchase Amount"].sum()), 2) if not last_7.empty else 0.0,
                    "last_30d_sales": round(float(last_30["Total Purchase Amount"].sum()), 2) if not last_30.empty else 0.0,
                }
        except Exception:
            sales = None

        # If OpenAI is available, use it; otherwise use heuristic reply
        if _openai_client and _OPENAI_KEY:
            context = _build_context_summary(product_info, reviews, sales)
            messages = [{"role": "system", "content": "You are a concise product expert. Summarize products from e-commerce links, explain use-cases, ask clarifying questions, and assess fit based on user's needs. Include recent sales context when available."}]
            if context:
                messages.append({"role": "system", "content": context})
            for h in history:
                if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
                    messages.append({"role": h["role"], "content": h["content"]})
            if message:
                messages.append({"role": "user", "content": message})

            try:
                completion = _openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=messages,
                    temperature=0.3,
                )
                reply_text = completion.choices[0].message.content.strip()
                return jsonify({
                    "reply": reply_text,
                    "used_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                })
            except Exception as oe:
                # Fallback to heuristic on any OpenAI failure
                reply_text = _heuristic_ai_reply(message, product_info, reviews, sales)
                return jsonify({
                    "reply": reply_text,
                    "used_model": "heuristic-fallback",
                    "error": f"OpenAI error: {str(oe)}"
                })
        else:
            reply_text = _heuristic_ai_reply(message, product_info, reviews, sales)
            return jsonify({
                "reply": reply_text,
                "used_model": "heuristic-fallback"
            })
    except Exception as e:
        return jsonify({"error": f"AI assist failed: {str(e)}"}), 500


def scrape_amazon(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)

            title = page.query_selector("#productTitle")
            image = page.query_selector("#landingImage")
            price = page.query_selector(".a-price .a-offscreen")
            reviews = page.query_selector_all(".review-text-content span")

            review_texts = [rev.inner_text().strip() for rev in reviews[:5]] if reviews else ["No reviews found"]
            sentiment_score = compute_sentiment_score(review_texts)

            product_info = {
                "title": title.inner_text().strip() if title else "No title found",
                "image": image.get_attribute("src") if image else "",
                "price": price.inner_text().strip() if price else "N/A",
                "rating": "N/A",
                "source": "Amazon",
                "sentiment_score": sentiment_score 
            }

            browser.close()
            return {
                "product_info": product_info,
                "reviews": review_texts
            }

    except Exception as e:
        print("Error scraping Amazon:", e)
        return {"error": "Failed to fetch Amazon product details."}

def scrape_flipkart(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)

            title = page.query_selector("span.B_NuCI")
            image = page.query_selector("img._396cs4")
            price = page.query_selector("div._30jeq3._16Jk6d")
            reviews = page.query_selector_all("div.t-ZTKy div")

            review_texts = [rev.inner_text().strip() for rev in reviews[:5]] if reviews else ["No reviews found"]
            sentiment_score = compute_sentiment_score(review_texts)
            
            product_info = {
                "title": title.inner_text().strip() if title else "No title found",
                "image": image.get_attribute("src") if image else "",
                "price": price.inner_text().strip() if price else "N/A",
                "rating": "N/A",
                "source": "Flipkart",
                "sentiment_score": sentiment_score 
            }

            browser.close()
            return {
                "product_info": product_info,
                "reviews": review_texts
            }

    except Exception as e:
        print("Error scraping Flipkart:", e)
        return {"error": "Failed to fetch Flipkart product details."}

# ---- Run Flask App ----
if __name__ == "__main__":
    app.run(debug=True)
