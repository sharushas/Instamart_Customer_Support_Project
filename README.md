# Instamart Customer Intelligence & Support Copilot

## Overview

Swiggy Instamart delivers groceries, dairy, snacks, personal-care items, and small electronics across six metro cities within minutes. As order volumes scale, leadership encounters three recurring challenges:
1. **Data Distrust**: Raw order exports contain duplicates, inconsistent naming, pending records, and impossible values (negative delivery times, corrupted amounts).
2. **Customer Churn Uncertainty**: Retention teams lack early warning indicators to distinguish active shoppers from customers about to lapse.
3. **Support Desk Overload**: Thousands of unstructured free-text tickets flood customer care, mixing billing refunds, fraud alerts, technical errors, and delivery delays without automated triage or privacy protection.

This repository implements the complete end-to-end **Customer Intelligence & Support Copilot** built across four interconnected parts:
- **Part A (Data Foundation)**: Cleans, audits, and structures raw exports into a trusted SQLite database (`data/instamart.db`) and evaluates business metrics through analytical SQL and pandas.
- **Part B (Customer Segments & Lapse Prediction)**: Derives an RFM segmentation (K-Means K=4, PCA), defines a strictly leakage-free churn prediction dataset, and trains tuned classification models (Random Forest and Logistic Regression) with class-balancing experiments.
- **Part C (Support Ticket Triage with Guardrails)**: Implements privacy guardrails (PII redaction), input token bounds, resilient exponential backoff with jitter against transient server failures, rule-based output validation, and intelligent customer-segment routing.
- **Part D (Policy RAG Assistant & Refund ReAct Agent)**: Implements a TF-IDF grounded policy knowledge retrieval system and an autonomous ReAct refund agent that executes database queries, enforces policy limits, maintains state checkpoints, supports crash recovery, and guarantees graceful fallbacks.

---

## Reproduce

Follow these numbered shell commands in order from a clean terminal to reproduce all checkpoints, database tables, models, and outputs:

```bash
# 1. Create isolated Python virtual environment
python -m venv .venv

# 2. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Or on Linux / macOS: source .venv/bin/activate

# 3. Install dependencies from requirements.txt
pip install -r requirements.txt

# 4. Generate synthetic raw data and policy documents
python generate_data.py

# 5. Run Part A: Clean raw data, populate SQLite database, and export clean CSVs
python src/part_a_pipeline.py

# 6. Validate Part A reference checkpoints
python src/validate_checkpoints.py

# 7. Execute analytical SQL queries against data/instamart.db
python src/run_sql_queries.py

# 8. Run Part A Exploratory Data Analysis & pandas cross-checks
python src/part_a_eda.py

# 9. Run Part B: RFM segmentation, K-Means clustering, PCA, and lapse model tuning
python src/part_b_pipeline.py

# 10. Run Part C: Ticket triage, PII redaction, prompt comparison, budget demo, and routing
python src/part_c_pipeline.py

# 11. Run Part D: Policy RAG evaluation, ReAct refund scenarios, and crash-resume test
python src/part_d_pipeline.py

# 12. Run the interactive Customer Intelligence & Support Copilot CLI
python copilot.py
```

### Interactive Customer Intelligence & Support Copilot (`copilot.py`)
The terminal Copilot application (`copilot.py`) unifies all four project components into an interactive console tool:
1. **Guided Sequential Tour**: Sequentially executes Parts A through D with interactive step approval.
2. **Policy RAG Assistant**: Interactively queries the 15-chunk TF-IDF knowledge base over company policies.
3. **Autonomous Refund ReAct Agent**: Live planning and execution of database lookups, policy validation, and refund issuance/escalation.
4. **Support Ticket Triage**: Live input sanitization (Email $\rightarrow$ Phone $\rightarrow$ Order ID), token budget bounds, few-shot classification, and segment routing.
5. **Customer Intelligence & Churn Predictor**: Retrieves RFM statistics and calculates machine-learning churn risk probabilities from `outputs/lapse_model.joblib`.
6. **Smart Conversational Copilot**: Multi-intent natural language understanding with automatic disambiguation between policy inquiries, customer profile queries, refund requests, and support tickets.

---

## Part A — Data Foundation: Clean, Store, Query

### 1. Raw Data Audit
The raw files were loaded into pandas with `keep_default_na=False` on `tickets.csv`. The audit revealed severe anomalies across customers, orders, and tickets:

| Table | Total Rows | Columns | Identified Data Quality Issues |
| :--- | :---: | :---: | :--- |
| `orders.csv` | 13,084 | 9 | 119 exact duplicates, 656 pending records, 120 non-numeric amounts, 106 negative delivery times, 2,541 blank delivery times |
| `customers.csv` | 1,200 | 5 | 61 blank ages, 6 impossible ages outside [10, 100], 24 messy city spellings, 10 non-blank gender variants, 30 blank genders |
| `products.csv` | 40 | 4 | 0 missing values, valid catalog |
| `tickets.csv` | 164 | 5 | 4 empty tickets, 4 extremely long tickets (~520 words, 734 estimated tokens) |

#### Column-by-Column Schema & Missing Values Audit
| Table | Column Name | Raw Data Type | Missing Count | Missing Percentage | Audit Finding / Anomaly |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`customers.csv`** | `customer_id` | `object` (string) | 0 | 0.00% | Unique identifier C0001–C1200 |
| | `city` | `object` (string) | 0 | 0.00% | 24 messy casing/space variants |
| | `signup_date` | `object` (string) | 0 | 0.00% | ISO date strings YYYY-MM-DD |
| | `age` | `object` (string) | 61 | 5.08% | 61 blank + 6 impossible ages (-4, 0, 150, 200, 280, 999) |
| | `gender` | `object` (string) | 30 | 2.50% | 30 blank + 10 distinct casing variants |
| **`products.csv`** | `product_id` | `object` (string) | 0 | 0.00% | P001–P040 |
| | `product_name` | `object` (string) | 0 | 0.00% | Catalog item descriptions |
| | `category` | `object` (string) | 0 | 0.00% | 5 categories: Grocery, Dairy, Snacks, Personal Care, Electronics |
| | `price_inr` | `int64` | 0 | 0.00% | Integer prices in INR (20 to 1999) |
| **`orders.csv`** | `order_id` | `object` (string) | 0 | 0.00% | 12,309 unique IDs; 119 exact duplicates + 656 pending duplicates |
| | `customer_id` | `object` (string) | 0 | 0.00% | Customer mapping |
| | `product_id` | `object` (string) | 0 | 0.00% | Product catalog foreign key |
| | `quantity` | `int64` | 0 | 0.00% | Order quantity (1 to 3 items) |
| | `order_ts` | `object` (string) | 0 | 0.00% | Timestamp YYYY-MM-DD HH:MM:SS |
| | `status` | `object` (string) | 0 | 0.00% | completed, cancelled, pending |
| | `discount_pct` | `int64` | 0 | 0.00% | Promotion discount (0% to 40%) |
| | `order_amount` | `object` (string) | 0 | 0.00% | 120 unparseable corrupted entries ('N/A', 'error') |
| | `delivery_minutes` | `object` (string) | 2,541 | 19.42% | 2,541 blank + 106 impossible negative (-10) durations |
| **`tickets.csv`** | `ticket_id` | `object` (string) | 0 | 0.00% | T0001–T0164 |
| | `customer_id` | `object` (string) | 0 | 0.00% | Customer mapping |
| | `order_id` | `object` (string) | 8 | 4.88% | 8 blank (4 empty tickets + 4 long tickets) |
| | `text` | `object` (string) | 4 | 2.44% | 4 completely empty ticket texts |
| | `true_category` | `object` (string) | 0 | 0.00% | billing, delivery, technical, fraud, none |

#### Specific Audit Checkpoints
- **Raw orders rows**: 13,084
- **Exact duplicate rows**: 119
- **Pending rows**: 656
- **Distinct `order_id` values**: 12,309
- **Non-numeric `order_amount` entries**: 120
- **Negative `delivery_minutes`**: 106
- **Blank `delivery_minutes`**: 2,541
- **Blank customer `age`**: 61
- **Customer `age` outside 10–100**: 6 (values: -4, 0, 150, 200, 280, 999)
- **Distinct raw city spellings**: 24 (mixed uppercase, lowercase, whitespace padding)
- **Distinct non-blank raw gender spellings**: 10 (`male`, `Male`, `M`, `m`, `MALE`, `female`, `Female`, `F`, `f`, `FEMALE`)
- **Blank customer `gender`**: 30

---

### 2. Cleaning Rules & Transformations

#### Customer Table Cleaning
1. **City Standardization**: Stripped leading/trailing whitespace and converted to title case. Exactly 6 metro cities remain: Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184).
2. **Gender Mapping**: Standardized variants into `male` (560), `female` (610), and mapped blank/missing values to `unknown` (30).
3. **Age Imputation**: Replaced 6 impossible values outside [10, 100] and 61 missing values with `np.nan`. Computed the median age per city using `groupby('city')['age'].transform('median')` and filled missing entries. Result: 0 missing ages; every age is an integer between 18 and 60.
4. **Signup Date**: Parsed into datetime format `YYYY-MM-DD`.

#### Orders Table Cleaning
1. **Timestamp Conversion**: Converted `order_ts` to datetime format.
2. **Exact Duplicate Removal**: Dropped 119 redundant rows via `drop_duplicates()`.
3. **Status Deduplication**: For orders appearing multiple times (e.g., initial `pending` followed by `completed` or `cancelled`), sorted by `order_ts` and kept strictly the latest record per `order_id` via `drop_duplicates(subset=['order_id'], keep='last')`. This removed all 656 pending rows and resolved order states to 12,309 unique orders.
4. **Amount Repair**: Coerced unparseable amounts (`N/A`, `error`) to missing (120 values) and recalculated from product catalog price:
   $$\text{order\_amount} = \text{round}(\text{price\_inr} \times \text{quantity} \times (1 - \text{discount\_pct}/100), 2)$$
   Zero missing amounts remain.
5. **Delivery Time Imputation**: Set 106 negative delivery minutes to `np.nan`. For orders with status `completed`, filled missing delivery times with the median delivery duration of that product category among completed orders using `groupby('category')['delivery_minutes'].transform('median')`. For `cancelled` orders, delivery minutes were preserved as missing (`np.nan`) because cancelled orders were never dispatched.

#### Post-Cleaning Reference Verification
- Clean orders count: **12,309** (one per `order_id`)
- Order statuses: **11,110 completed**, **1,199 cancelled**, **0 pending**
- Missing `order_amount`: **0**
- Missing `delivery_minutes`: **0** among completed orders; **1,199** among cancelled orders
- Total completed revenue: **INR 2,998,668.80**
- Checkpoint validation script (`src/validate_checkpoints.py`): prints `ALL CHECKPOINTS PASSED`.

---

### 3. Analytical SQL Query Results (`sql/queries.sql`)

All queries were executed via `pd.read_sql_query` against `data/instamart.db`:

- **Q1 (WHERE completed & amount > 500, GROUP BY customer HAVING total > 8000)**: Returned **30 customers**.
- **Q2 (Three-table join by city and category)**: Returned **30 rows** with total revenue of **INR 2,998,668.80**. The largest revenue cell is **Bengaluru / Electronics** at **INR 233,909.45**.
- **Q3 (Scalar subquery for products priced above catalog average)**: Returned **11 products** (average catalog price: INR 377.92).
- **Q4 (LEFT JOIN for customers with no completed order)**: Returned **25 customers**.
- **Q5 (Common Table Expression for spend above average)**: Returned **550 customers** spending above the customer average of **INR 2,552.06**.
- **Q6 (Ranking window functions by city spend)**: Top customers per city:
  - **Bengaluru**: C0082 (INR 12,856.20)
  - **Chennai**: C0930 (INR 10,151.40)
  - **Delhi**: C0291 (INR 13,742.60)
  - **Hyderabad**: C0743 (INR 12,730.05)
  - **Mumbai**: C0693 (INR 14,766.55)
  - **Pune**: C0426 (INR 10,377.10)
  - Disparities: Exactly **8 rows** have `RANK()` different from `ROW_NUMBER()`.
  - **Ties explanation**: Ties in total completed spend receive identical `RANK()` values (e.g., sharing rank 45), whereas `ROW_NUMBER()` arbitrarily forces sequential numbering without duplicates (e.g., 45 and 46), causing rank disparities whenever ties occur.
- **Q7 (Running total of daily completed revenue)**: Returned **183 daily rows** spanning 2025-04-01 through 2025-09-30, with final cumulative revenue of **INR 2,998,668.80**.

---

### 4. pandas Cross-Check & Exploratory Data Analysis

#### City Revenue Breakdown
| City | Completed Orders Count | Mean Completed Order Amount (INR) | Total Revenue (INR) |
| :--- | :---: | :---: | :---: |
| **Bengaluru** | 1,894 | 276.47 | 523,643.40 |
| **Hyderabad** | 1,855 | 284.95 | 528,589.90 |
| **Delhi** | 2,064 | 243.62 | 502,829.80 |
| **Pune** | 1,984 | 252.48 | 500,918.05 |
| **Mumbai** | 1,613 | 297.00 | 479,058.50 |
| **Chennai** | 1,700 | 272.72 | 463,629.15 |
| **All (Total)** | **11,110** | **269.91** | **2,998,668.80** |

#### Distribution Statistics for Completed Order Amount
- **Mean**: INR 269.91
- **Median**: INR 135.00
- **Skewness**: 4.61 (heavily right-skewed due to high-value electronics)
- **Q1 (25th percentile)**: INR 64.80
- **Q2 (50th percentile)**: INR 135.00
- **Q3 (75th percentile)**: INR 297.00
- **Interquartile Range (IQR)**: INR 232.20
- **Lower Fence ($Q_1 - 1.5 \times \text{IQR}$)**: -INR 283.50 (0 outliers below)
- **Upper Fence ($Q_3 + 1.5 \times \text{IQR}$)**: INR 645.30 (897 outliers above)

#### Correlation Matrix
| Metric | `order_amount` | `discount_pct` | `quantity` | `delivery_minutes` |
| :--- | :---: | :---: | :---: | :---: |
| **`order_amount`** | 1.0000 | -0.0951 | 0.0873 | -0.0031 |
| **`discount_pct`** | -0.0951 | 1.0000 | -0.0004 | -0.0098 |
| **`quantity`** | 0.0873 | -0.0004 | 1.0000 | -0.0126 |
| **`delivery_minutes`** | -0.0031 | -0.0098 | -0.0126 | 1.0000 |

#### Exploratory Findings (Under 200 words)
This dataset tracks rapid grocery and electronics fulfillment across six Indian metros over six months. Order values exhibit extreme right skewness (mean INR 269.91 vs. median INR 135.00, skewness 4.61). The lower 75% of purchases consist of inexpensive staples, snacks, and dairy (< INR 297.00), while high-value electronics (smartphones, audio gear up to INR 1,999.00) produce 897 upper-fence outliers beyond INR 645.30. Delivery times remain tightly centered around 24 minutes, with rare operational delays extending beyond 45 minutes. The strongest pairwise correlation in absolute value is between order amount and discount percentage at -0.0951 (with quantity slightly lower at +0.0873); however, a correlation of this minor magnitude does not establish causation, reflecting synthetic promotional variability rather than price elasticity.

---

## Part B — Customer Segments and Lapse Prediction

### 1. RFM Table
Calculated at the snapshot date **2025-10-01 00:00:00** over all 1,175 customers with completed orders:
- **Customer Count**: 1,175
- **Recency (days)**: Min = 0, Median = 19.0, Max = 180
- **Frequency (completed orders)**: Mean = 9.46, Max = 33
- **Monetary (completed spend)**: Mean = INR 2,552.06, Max = INR 14,766.55

---

### 2. K-Means Clustering Evaluation
RFM features were standardized using `StandardScaler`. Clustering was evaluated across $K \in [2, 8]$:

| K | Inertia | Silhouette Score | Evaluation Notes |
| :---: | :---: | :---: | :--- |
| 2 | 1820.8 | 0.465 | High within-cluster variance |
| 3 | 1268.2 | 0.445 | Coarse grouping |
| **4** | **743.1** | **0.526** | **Global peak in silhouette score; distinct elbow point** |
| 5 | 617.4 | 0.480 | Diminishing separation |
| 6 | 503.2 | 0.482 | Segment fragmentation |
| 7 | 408.5 | 0.416 | Overfitting |
| 8 | 366.5 | 0.412 | Excessive subdivision |

*Why inertia alone cannot choose K*: Adding cluster centroids mechanically increases model degrees of freedom, guaranteeing that inertia monotonically decreases until reaching zero at $K=N$. Because lower inertia reflects fitting complexity rather than generalizable structure, selection requires identifying an elbow plateau, peak silhouette width, and business interpretability.

---

### 3. Customer Segments Profile (Chosen K=4)
Cluster IDs are arbitrary identifiers with no inherent ordering. Segments were profiled, named, and mapped to strategic actions:

| Segment ID | Segment Name | Customer Count | Mean Recency (days) | Mean Frequency (orders) | Mean Monetary (INR) | Recommended Business Action |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Premium Shoppers ⭐** | **138** | 24.1 | 6.8 | **INR 7,129.70** | Priority concierge SLA, electronics protection warranties (`is_premium = True`) |
| **2** | **Everyday Loyalists** | **419** | 8.8 | 19.4 | INR 3,410.02 | Subscription delivery pass (Swiggy One), basket-building gamification |
| **0** | **Occasional Value Seekers** | **436** | 28.9 | 3.8 | INR 1,057.62 | Push cart abandonment discounts, targeted weekend meal-kit promos |
| **3** | **At-Risk / Lapsed** | **182** | 109.0 | 2.2 | INR 685.99 | Aggressive reactivation vouchers (INR 150 off), feedback surveys |

- **PCA 2D Projection**: Applied `PCA(n_components=2)`. Explained variance ratios are **0.634 (PC1)** and **0.218 (PC2)**, capturing **85.2% of total variance**.
- Exported to `outputs/customer_segments.csv`.

---

### 4. Lapse Dataset Construction & Leakage Prevention

- **Cutoff Moment**: `2025-09-01 00:00:00`.
- **Target Definition (`lapsed`)**: A customer in the cohort is labeled $1$ if they placed **zero completed orders** between 2025-09-01 and 2025-09-30, and $0$ otherwise.
- **Strict Leakage Prevention Guarantee**:
  - No orders dated on or after 2025-09-01 were used in feature generation.
  - Segment labels from Task 3 were not used as features.
  - All preprocessing transformers (encoders, scalers) were fitted exclusively on training rows.
- **Cohort Size**: 1,156 customers with $\ge 1$ completed order before cutoff.
- **Lapse Distribution**: 443 lapsed (38.32%), 713 active (61.68%).
- **Split**: Stratified 80/20 train/test split (`random_state=42`) producing **924 training rows** (570 active, 354 lapsed) and **232 test rows** (143 active, 89 lapsed).
- **Baseline Model**: Always predicting "Not Lapsed" ($0$) yields **Accuracy: 61.6%**, **Lapsed F1: 0.0000**.

---

### 5. Preprocessing & Leakage Check
- **Feature Pipeline**: Nominal columns (`city`, `gender`, `favourite_category`) encoded using `OneHotEncoder(drop='first', sparse_output=False)`; numeric features standardized via `StandardScaler()`.
- **Frequency Learned Mean Comparison**:
  - Training Set Mean: **8.1147**
  - Full Cohort Mean: **8.0285**
  - *Reason for difference*: The training split represents an 80% stratified subsample whose sample mean naturally diverges slightly from the population mean due to random sampling variance.
- Missing values in final feature matrix: **0**.

---

### 6. Model Tuning & Test Set Evaluation

Both models were tuned using 5-fold cross-validation on training rows:
- **Logistic Regression**: Grid over $C \in \{0.01, 0.1, 1, 10, 100\}$. Best $C=100$, Best CV F1: **0.6506**.
- **Random Forest**: Grid over `max_depth` $\in \{3, 5, 8\}$ and `min_samples_leaf` $\in \{5, 10, 20\}$ (45 fits). Best parameters: `max_depth=8`, `min_samples_leaf=5`, Best CV F1: **0.6848**.

#### Test Set Performance Comparison (Lapsed Positive Class)
| Model | Test Accuracy | Lapsed Precision | Lapsed Recall | Lapsed F1 | Confusion Matrix (TN, FP / FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Majority Class)** | 61.6% | 0.0000 | 0.0000 | 0.0000 | `[[143, 0], [89, 0]]` |
| **Tuned Logistic Regression** | 72.8% | **0.6444** | 0.6517 | **0.6480** | `[[111, 32], [31, 58]]` |
| **Tuned Random Forest** | 72.0% | 0.6304 | 0.6517 | **0.6409** | `[[109, 34], [31, 58]]` |

#### Model Selection Rationale
Random Forest was selected as the operational deployment model (`outputs/lapse_model.joblib`). In customer retention economics, the business cost of a false negative (failing to identify an actually churning customer) leads to permanent customer lifetime value loss, which is vastly more expensive than a false positive (sending an automated retention coupon to a customer who would have ordered anyway). Random Forest captures high-order non-linear interactions between delivery delays, product categories, and purchase recency, providing reliable risk calibration.

---

### 7. Class-Balance Experiment
The training set (570 non-lapsed, 354 lapsed) was oversampled by randomly duplicating lapsed rows with replacement (`random_state=42`) until reaching parity (570 in each class, 1,140 total rows). The model was refit and evaluated on the **unchanged, original test set**:

| Training Regime | Test Precision | Test Recall | Test F1-Score | Lapsed Detected |
| :--- | :---: | :---: | :---: | :---: |
| **Unbalanced Logistic Regression** | **0.6444** | 0.6517 | 0.6480 | 58 / 89 |
| **Oversampled Balanced Logistic Regression** | 0.5891 | **0.8539** | **0.6972** | **76 / 89** |

*Key finding for retention team*: **Recall moved most dramatically**, surging from **0.6517 to 0.8539** (+20.2 percentage points). Precision dipped slightly from 0.6444 to 0.5891. For the retention team, this trade-off is highly advantageous: the balanced model successfully catches **85.4% of all churning customers** while maintaining acceptable promotional efficiency.

---

### 8. Interpretability & Retention Prioritization

#### Top 5 Random Forest Features (`feature_importances_`)
1. **`monetary` (0.2591)**: Total spend before cutoff is the primary discriminator of customer commitment and habit.
2. **`frequency` (0.2546)**: Order volume reflects daily replenishment habits; high frequency strongly buffers against churn.
3. **`recency` (0.1080)**: Days since last order directly indicates declining brand engagement.
4. **`avg_delivery_minutes` (0.0756)**: Chronic delivery delays trigger customer dissatisfaction and abandonment.
5. **`avg_discount_pct` (0.0568)**: Heavy promotion reliance indicates deal-seeking behaviour that lapses once discounts cease.

#### Logistic Regression Coefficients
- **`recency` (+0.0396)**: Positive coefficient confirms that longer gaps since the last purchase increase churn odds.
- **`frequency` (-1.6982)**: Strong negative coefficient confirms that regular purchase routines dramatically decrease lapse likelihood.
- **`monetary` (+0.2846)**: Positive coefficient (holding frequency constant) captures single-purchase high-ticket buyers (electronics) who lack recurring grocery needs.

#### Top 10 High-Risk Premium Customers (`outputs/retention_list.csv`)
| Rank | Customer ID | Predicted Lapse Probability | Actually Lapsed (Sept 2025) | Segment |
| :---: | :---: | :---: | :---: | :---: |
| 1 | **C0862** | **0.5623** | 0 | Premium Shoppers |
| 2 | **C0167** | **0.5258** | 0 | Premium Shoppers |
| 3 | **C1112** | **0.5194** | 0 | Premium Shoppers |
| 4 | **C1008** | **0.5095** | 0 | Premium Shoppers |
| 5 | **C0133** | **0.5093** | 0 | Premium Shoppers |
| 6 | **C0060** | **0.4903** | 0 | Premium Shoppers |
| 7 | **C0267** | **0.4841** | 0 | Premium Shoppers |
| 8 | **C1000** | **0.4836** | 0 | Premium Shoppers |
| 9 | **C0082** | **0.4817** | 0 | Premium Shoppers |
| 10 | **C0682** | **0.4807** | **1** | Premium Shoppers |

*Methodological Note*: The Premium segment flag originates from the whole-window RFM analysis (which includes the September orders defining the target), making this retention list illustrative rather than a leakage-free evaluation.

---

## Part C — Support Ticket Triage with Guardrails

### 1. Privacy Guardrails & PII Redaction Audit
Across all 164 tickets in `data/raw/tickets.csv`, regular expressions were applied in strict sequential order:
1. Email addresses: `[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]` (**55 substitutions**)
2. Phone numbers: `\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]` (**91 substitutions**)
3. Order IDs: `ORD-\d{6}` $\rightarrow$ `[ORDER_ID]` (**132 substitutions**)
- **Total PII substitutions**: **278**
- **Residual PII check**: Re-running all three regexes over the redacted corpus yields **0 matches**.

---

### 2. Resilient Retries & Error Handling
Calls to `mock_llm.chat` with `mock_llm.reset()` were wrapped in an exponential backoff function with jitter:
- Formula: $\text{wait} = 0.05 \times 2^n \times \text{uniform}(0.9, 1.15)$ seconds for retry attempt $n \in \{0, 1, 2\}$.
- Across the 156 eligible tickets, exactly **19 `TransientLLMError` ("503 server busy") exceptions** were triggered (every 9th call).
- All 19 failures were successfully recovered on retry attempt 0 with wait durations between **0.0450s and 0.0562s**. No permanent errors occurred.

---

### 3. Output Validation Breakdown
Every model response was validated under strict safety rules:
1. **Valid JSON with allowed category** (`classified_json`): **135 tickets**
2. **Chatty non-JSON parsed via word extraction** (`classified_keyword`): **12 tickets** (e.g., *"The customer appears to have a billing-related problem."* parsed via `re.findall(r"[a-z]+", text.lower())`).
3. **Invalid output routed to human review** (`flagged_invalid_output`): **9 tickets** (simulated bad JSON or illegal categories like `shipping`).
4. **Input guardrail stops**: **4 empty tickets** (`skipped_empty`) and **4 long tickets** (`flagged_long`, 734 estimated tokens).

---

### 4. Zero-Shot vs Few-Shot Prompt Comparison

| Prompt Strategy | Classified Tickets | Classification Accuracy | Total Tokens Consumed | Tokens per Classified Ticket |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-Shot** | 147 | **85.7%** (126 / 147) | 12,520 | **85.2** |
| **Few-Shot** | 147 | **100.0%** (147 / 147) | 29,212 | **198.7** |

#### Zero-Shot Confusion Matrix
| True Category | Predicted `billing` | Predicted `delivery` | Predicted `fraud` | Predicted `technical` |
| :--- | :---: | :---: | :---: | :---: |
| **billing** | **38** | 0 | 0 | 0 |
| **delivery** | 0 | **36** | 0 | 0 |
| **fraud** | **11** | 0 | **23** | 0 |
| **technical** | **10** | 0 | 0 | **29** |

*When few-shot tokens are worth paying for*: Few-shot prompting requires **198.7 tokens/ticket versus 85.2 tokens/ticket** (+133% token overhead). However, zero-shot classification severely misclassifies fraud (11 tickets) and technical issues (10 tickets) as billing due to overlapping terms ("charged", "payment", "card"). In an operational setting, misrouting fraud to regular billing queues violates the mandatory 2-hour security SLA. Paying the extra token cost is essential to eliminate misrouted critical escalations.

---

### 5. Token Budgeting & Routing

- **Token Budget Demo (`TOKEN_BUDGET = 6000`)**: Running zero-shot classification halted after processing 83 tickets; exactly **81 tickets** ended with status `budget_exhausted` and were routed to human review (`outputs/triage_budget_demo.csv`).
- **Final Triage Routing (`outputs/triage_results.csv`, 164 tickets)**:
  - `support_queue` (Priority NORMAL): **102 tickets**
  - `security_team` (Priority HIGH, fraud): **34 tickets**
  - `human_review` (Priority NORMAL, empty/long/invalid/exhausted): **17 tickets**
  - `premium_desk` (Priority HIGH, premium segment): **11 tickets**

---

## Part D — Policy RAG Assistant and Refund Agent

### 1. Document Chunking & Vectorization
The five policy files were split into 40-word chunks with a 10-word overlap (step size: 30 words):
- `refund_policy.txt`: **4 chunks**
- `delivery_policy.txt`: **3 chunks**
- `escalation_policy.txt`: **3 chunks**
- `payment_policy.txt`: **3 chunks**
- `privacy_policy.txt`: **2 chunks**
- **Total Chunks**: **15 chunks**

*TF-IDF Theory & Limitations*: Term Frequency (TF) measures how often a word occurs within an individual chunk, while Inverse Document Frequency (IDF) scales down ubiquitous words across all documents to amplify rare, distinctive terms. A primary limitation is the lack of semantic awareness: TF-IDF cannot recognize conceptual equivalence between synonyms (e.g., "delayed" vs. "late") without exact token stem overlap.

*Verification*: Querying `"refund policy late delivery"` returned top-3 chunks exclusively from `refund_policy.txt`.

---

### 2. Policy Assistant Evaluation (`policy_eval.csv`)

| Chunking Configuration | Expected Doc in Top-3 | Fact Contained in Answer | Unanswerable Fallback Rate |
| :--- | :---: | :---: | :---: |
| **Standard (40 words, Overlap 10)** | **12 / 12** (100%) | **12 / 12** (100%) | **2 / 2** (100%) |
| **Zero Overlap (40 words, Overlap 0)** | 12 / 12 (100%) | **10 / 12** (83.3%) | 2 / 2 (100%) |

#### Analysis of Broken Question under Zero Overlap
- **Question**: *"What delivery time makes a completed order eligible for a full refund?"*
- **Expected Fact**: `"more than 45 minutes"`
- **Returned Answer**: *"receives a full refund of the order amount."*
- **Root Cause**: Without overlap, the critical clause boundary split between words 39 and 40 (*"If a completed order is delivered in"* in chunk 1, and *"more than 45 minutes, the customer receives..."* in chunk 2), depriving the retriever of contiguous context.

---

### 3. ReAct Refund Agent Seven-Scenario Results

The autonomous refund agent was evaluated across all seven scenarios from `data/raw/agent_scenarios.csv`:

| Scenario ID | Final Status | Refund Amount (INR) | Expected Outcome | Expected Amount (INR) | Model Calls | Tokens Used | Step Summary |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **S1** | **REFUND** | **128.25** | REFUND | 128.25 | 5 | 895 | Completed order >45 min; approved automatically |
| **S2** | **REFUND** | **135.00** | REFUND | 135.00 | 5 | 895 | Completed order >45 min; approved automatically |
| **S3** | **ESCALATE** | **1999.00** | ESCALATE | 1999.00 | 5 | 882 | Amount > INR 1000; sent to human support lead |
| **S4** | **DENIED** | **0.00** | DENIED | 0.00 | 4 | 601 | Delivered in $\le 45$ min; late refund denied |
| **S5** | **DENIED** | **0.00** | DENIED | 0.00 | 3 | 334 | Order status cancelled; not eligible for refund |
| **S6** | **NOT_FOUND** | **0.00** | NOT_FOUND | 0.00 | 2 | 70 | Non-existent order id in database |
| **S7** | **FALLBACK_MAX_ITERATIONS** | **0.00** | FALLBACK | 0.00 | 6 | 2,994 | Loop prompt; halted gracefully after 6 calls |

All ledger-modifying actions were logged to `outputs/refund_ledger.jsonl`. Complete trajectory traces were preserved in `outputs/agent_runs.jsonl`.

---

### 4. Stop Safely Demonstrations

1. **Token Limit Guardrail (`token_limit = 200`)**: Evaluated on S1. The agent stopped immediately after the first model call without executing the requested tool, recording status **`FALLBACK_TOKEN_LIMIT`**.
2. **Timeout Guardrail (`time_limit_seconds = 0`)**: Evaluated on S1. The agent stopped before initiating the first model call, recording status **`FALLBACK_TIMEOUT`**.

---

### 5. Crash and Resume Demonstration

- **Simulated Crash**: Triggered during scenario S1 immediately after the second tool execution (`search_policy`), saving the state checkpoint to `state/<run_id>.json`.
- **Resumption**: Calling `resume_agent(run_id)` loaded the checkpoint and completed execution.
- **Tools Executed Across Crash and Resume**:
  1. `get_order` (executed pre-crash)
  2. `search_policy` (executed pre-crash)
  3. `calculate_refund` (executed post-resume)
  4. `issue_refund` (executed post-resume)
- **Result**: Exactly **4 unique tools executed** with zero duplicate invocations; final status was **`REFUND`** with **INR 128.25** issued.

---

### 6. ReAct versus Plan-and-Execute Architecture (Under 250 words)

A plan-and-execute architecture prompts the language model to generate an entire sequence of tool invocations upfront in one call, executing sequentially and replanning only upon failure. In predictable workflows, this minimizes model calls and token consumption. However, for dynamic customer support and refund processing, execution paths depend entirely on intermediate observations:
1. S6 discovers `NOT_FOUND` in 2 calls and terminates immediately.
2. S5 identifies a cancelled order in 3 calls and denies the refund without computing amounts or searching policy.
3. S4 determines delivery was under 45 minutes in 4 calls and denies without human escalation.
4. S1 and S2 issue refunds in 5 calls, while S3 routes amounts exceeding INR 1,000 to human leads.

Under a plan-and-execute approach, the model would needlessly generate full refund plans for non-existent or cancelled orders, requiring repetitive replanning exceptions. The ReAct pattern evaluates each observation before deciding the subsequent step. This dynamic grounding prevents wasted tool executions, enforces strict threshold safety, and allows incremental checkpointing for seamless crash recovery. For these reasons, ReAct is the superior design choice for financial customer support workflows.

---

## Limitations and Next Steps

1. **Post-Hoc Whole-Window Segment Flag**: The `is_premium` flag in `outputs/customer_segments.csv` and `outputs/retention_list.csv` was computed over the entire 6-month order history (including September 2025). While feature inputs for model training strictly respected the pre-September cutoff, the retention list serves an illustrative purpose rather than a strictly leakage-free evaluation. *Next Step*: Compute historical time-windowed RFM segments snapshot-aligned strictly prior to the churn observation window.
2. **Lexical Retrieval Boundaries**: TF-IDF relies on exact keyword matching, occasionally struggling with vernacular queries or synonyms (e.g., "rider unreachable" vs. "undeliverable"). *Next Step*: Deploy dense vector embeddings (e.g., `text-embedding-3-small` or open-source sentence-transformers) paired with cosine similarity and hybrid BM25 reranking.
3. **Simulated LLM Environment**: All model completions were driven deterministically by `mock_llm.py`. *Next Step*: Connect to production LLM APIs with streaming responses and asynchronous worker queues (Celery/Redis).
