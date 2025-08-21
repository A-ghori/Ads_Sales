const express = require("express");
const bodyParser = require("body-parser");
const path = require("path");

const app = express();
const PORT = 5030;

// Middleware
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "public"))); // static HTML serve karega

// Model coefficients (Python se copy karke paste karo)
const coefficients = [0.05450927,0.10094536,0.00433665];
const intercept = 4.714126402214127;

// Prediction function
function predictSales(tv, radio, newspaper) {
  return intercept +
    coefficients[0] * tv +
    coefficients[1] * radio +
    coefficients[2] * newspaper;
}

// API route
app.post("/predict", (req, res) => {
  const { tv, radio, newspaper } = req.body;

  if (tv === undefined || radio === undefined || newspaper === undefined) {
    return res.status(400).json({ error: "Please provide tv, radio, and newspaper budgets." });
  }

  const prediction = predictSales(tv, radio, newspaper);
  res.json({ predicted_sales: prediction });
});

// Start server
app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
