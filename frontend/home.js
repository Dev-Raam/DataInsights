function updateSentimentBar(score) {
  const fill = document.getElementById("sentiment-bar-fill");
  const text = document.getElementById("sentiment-score-text");

  if (!fill || !text) return;

  if (!score || isNaN(score)) {
    fill.style.width = "0%";
    fill.style.backgroundColor = "#ccc";
    text.textContent = "No sentiment score available";
    return;
  }

  const percentage = (score / 5) * 100;
  fill.style.width = `${percentage}%`;
  text.textContent = `Score: ${score.toFixed(2)} / 5`;

  // Color logic
  if (score >= 4) {
    fill.style.backgroundColor = "#4caf50"; // green
  } else if (score >= 3) {
    fill.style.backgroundColor = "#ff9800"; // orange
  } else {
    fill.style.backgroundColor = "#f44336"; // red
  }
}

document.addEventListener("DOMContentLoaded", () => {
  console.log("Reviews & Feedback JS Loaded!");

  const BASE_URL = "http://127.0.0.1:5000";

  const productSearchBtn = document.getElementById("productSearchBtn");
  const productSearchInput = document.getElementById("productSearchInput");
  const productDisplay = document.getElementById("productDisplay");
  const feedbackDisplay = document.getElementById("feedbackDisplay");

  if (productSearchBtn && productSearchInput && productDisplay && feedbackDisplay) {
    productSearchBtn.addEventListener("click", async () => {
      const productUrl = productSearchInput.value.trim();

      if (!productUrl) {
        alert("Please enter a product URL.");
        return;
      }

      try {
        const response = await fetch(`${BASE_URL}/api/fetch_product_details`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ product_url: productUrl })
        });

        const data = await response.json();

        if (data.error) {
          productDisplay.innerHTML = `<p style="color: red;">${data.error}</p>`;
          feedbackDisplay.innerHTML = "<p>No feedbacks available.</p>";
          updateSentimentBar(null);
          return;
        }

        const { image, title, price, rating, source, sentiment_score } = data.product_info;

        productDisplay.innerHTML = `
          <div style="text-align: left;">
            <img src="${image}" alt="Product Image" style="width: 100%; max-width: 200px; border-radius: 8px; margin-bottom: 10px;" />
            <h4>${title}</h4>
            <p><strong>Price:</strong> ${price}</p>
            <p><strong>Rating:</strong> ${rating || 'N/A'}</p>
            <p><strong>Source:</strong> ${source || 'N/A'}</p>
            <a href="${productUrl}" target="_blank" style="color: #40a8cb;">View Product</a>
          </div>
        `;

        updateSentimentBar(sentiment_score);

        if (data.reviews && data.reviews.length > 0) {
          let reviewHtml = "<h4>Reviews:</h4><ul>";
          data.reviews.forEach(review => {
            reviewHtml += `<li>${review}</li>`;
          });
          reviewHtml += "</ul>";
          feedbackDisplay.innerHTML = reviewHtml;
        } else {
          feedbackDisplay.innerHTML = "<p>No feedbacks available 🙄</p>";
        }
      } catch (err) {
        console.error("Error fetching product info:", err);
        productDisplay.innerHTML = `<p style="color: red;">Something went wrong.</p>`;
        feedbackDisplay.innerHTML = "<p>Could not load feedback.</p>";
        updateSentimentBar(null);
      }
    });
  }
});
