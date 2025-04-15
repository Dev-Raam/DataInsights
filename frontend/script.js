document.addEventListener("DOMContentLoaded", function () {
  console.log("JavaScript Loaded!");

  // Caching DOM elements
  const searchBtn = document.getElementById("searchBtn");
  const searchInput = document.getElementById("searchInput");
  const searchResultsList = document.getElementById("searchResultsList");
  const totalPurchases = document.getElementById("totalPurchases");
  const activeUsers = document.getElementById("activeUsers");
  const feedbackScore = document.getElementById("feedbackScore");
  const purchaseTrendsChart = document.getElementById("purchaseTrendsChart");
  const churnResult = document.getElementById("churnResult");
  const segmentResult = document.getElementById("segmentResult");
  const recommendationsResult = document.getElementById("recommendationsResult");

  const BASE_URL = "http://127.0.0.1:5000";

  // Search Functionality
  searchBtn.addEventListener("click", function () {
    const query = searchInput.value.trim();
    if (!query) {
      alert("Please enter a search term.");
      return;
    }
    fetchSearchResults(query);
  });

  async function fetchSearchResults(query = "") {
    try {
        const response = await fetch(`${BASE_URL}/api/search?query=${encodeURIComponent(query)}`);
        const data = await response.json();

        searchResultsList.innerHTML = "";

        if (query && data.search_results.length) {
            // Display search results
            data.search_results.forEach((item) => {
                const resultItem = document.createElement("div");
                resultItem.classList.add("search-item");
                resultItem.innerHTML = `
                    <p><strong>Customer:</strong> ${item["Customer Name"]}</p>
                    <p><strong>Product:</strong> ${item["Product Category"]}</p>
                    <p><strong>Purchase Amount:</strong> $${item["Total Purchase Amount"]}</p>
                `;
                searchResultsList.appendChild(resultItem);
            });
        } else if (!query && data.all_products.length) {
            // Show all available products when no query is entered
            searchResultsList.innerHTML = `<h3>All Available Products:</h3>`;
            data.all_products.forEach(product => {
                const productItem = document.createElement("p");
                productItem.textContent = product;
                searchResultsList.appendChild(productItem);
            });
        } else {
            searchResultsList.innerHTML = `<div class="no-results"><p>No results found</p></div>`;
        }
    } catch (error) {
        console.error("Error fetching search results:", error);
    }
  }


  // Fetch Dashboard Metrics
  async function fetchDashboardMetrics() {
    try {
      const response = await fetch(`${BASE_URL}/api/dashboard-metrics`);
      const data = await response.json();
      totalPurchases.innerText = data.total_purchases;
      activeUsers.innerText = data.active_users;
      feedbackScore.innerText = data.feedback_score;
    } catch (error) {
      console.error("Error fetching dashboard metrics:", error);
    }
  }

  // Fetch EDA Charts
  const chartIds = {
    spending_distribution: "spendingChart",
    churn_vs_spending: "churnSpendingChart",
    customer_age_distribution: "ageDistributionChart",
    feature_correlation_heatmap: "correlationHeatmapChart",
    churn_confusion_matrix_rf: "churnMatrixRFChart",
    churn_confusion_matrix_xgb: "churnMatrixXGBChart",
    churn_feature_importance: "churnFeatureImportanceChart",
    customer_elbow_method: "elbowMethodChart",
    sentiment_distribution: "sentimentDistributionChart"
  };

  function fetchCharts() {
    Object.keys(chartIds).forEach((chartKey) => {
      const imgElement = document.getElementById(chartIds[chartKey]);
      if (imgElement) {
        imgElement.src = `${BASE_URL}/api/charts/${chartKey}`;
      }
    });
  }

  // Fetch Purchase Trends (Chart.js)
  async function fetchPurchaseTrends() {
    try {
      const response = await fetch(`${BASE_URL}/api/purchase-trends`);
      const data = await response.json();
      new Chart(purchaseTrendsChart.getContext("2d"), {
        type: "line",
        data: {
          labels: data.labels,
          datasets: [{
            label: "Monthly Sales",
            data: data.values,
            borderColor: "rgba(75, 192, 192, 1)",
            borderWidth: 2,
            fill: false
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false
        }
      });
    } catch (error) {
      console.error("Error fetching purchase trends:", error);
    }
  }

  // Fetch Data for D3.js Chart (Customer Segmentation)
  async function fetchCustomerSegmentation() {
    try {
      const response = await fetch(`${BASE_URL}/api/customer-segmentation`);
      const data = await response.json();

      const svg = d3.select("#d3-chart").append("svg").attr("width", 500).attr("height", 300);
      svg.selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", (_, i) => i * 60)
        .attr("y", (d) => 300 - d.value * 3)
        .attr("width", 50)
        .attr("height", (d) => d.value * 3)
        .attr("fill", "teal");

      svg.selectAll("text")
        .data(data)
        .enter()
        .append("text")
        .text((d) => d.label)
        .attr("x", (_, i) => i * 60 + 15)
        .attr("y", 290)
        .attr("fill", "black");
    } catch (error) {
      console.error("Error fetching customer segmentation data:", error);
    }
  }

  // Handle Churn Prediction
  document.getElementById("predictChurnBtn").addEventListener("click", async function () {
    try {
      const response = await fetch(`${BASE_URL}/predict_churn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          "Total Spending": parseFloat(document.getElementById("totalSpending").value),
          "Purchase Frequency": parseInt(document.getElementById("purchaseFrequency").value),
          "Return Rate": parseFloat(document.getElementById("returnRate").value),
          "model": document.getElementById("churnModelType").value
        })
      });
      const data = await response.json();
      churnResult.innerText = `Prediction: ${data.churn_prediction ? "Will Churn" : "No Churn"} (Probability: ${data.churn_probability.toFixed(2)})`;
    } catch (error) {
      console.error("Error predicting churn:", error);
    }
  });

  // Handle Customer Segmentation
  document.getElementById("predictSegmentBtn").addEventListener("click", async function () {
    try {
      const response = await fetch(`${BASE_URL}/predict_segment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          "Recency": parseInt(document.getElementById("recency").value),
          "Frequency": parseInt(document.getElementById("frequency").value),
          "Monetary": parseFloat(document.getElementById("monetary").value)
        })
      });
      const data = await response.json();
      segmentResult.innerText = `Predicted Segment: ${data.predicted_segment}`;
    } catch (error) {
      console.error("Error predicting segment:", error);
    }
  });

  // Handle Product Recommendations
  document.getElementById("recommendProductsBtn").addEventListener("click", async function () {
    try {
      const response = await fetch(`${BASE_URL}/recommend_products`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ "customer_id": parseInt(document.getElementById("customerId").value) })
      });
      const data = await response.json();
      recommendationsResult.innerText = `Recommended Products: ${data.recommended_products.join(", ")}`;
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    }
  });

  // Initial Fetch Calls
  fetchDashboardMetrics();
  fetchCharts();
  fetchPurchaseTrends();
  fetchCustomerSegmentation();
});
