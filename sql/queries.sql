-- sql/queries.sql
-- Instamart Customer Intelligence & Support Copilot
-- Analytical SQL queries on data/instamart.db

-- Q1: WHERE then HAVING
-- Keep only completed orders with order_amount > 500 (WHERE), group by customer,
-- and keep customers whose total of those orders exceeds 8000 (HAVING). Reference: 30 customers.
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_amount
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: Three-table join
-- Revenue by city and product category (join orders, customers, products).
-- Reference: 30 rows, revenues sum to INR 2,998,668.80, largest cell is Bengaluru / Electronics at INR 233,909.45.
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: Scalar subquery
-- Products priced above the average catalogue price. Reference: 11 products.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN
-- Customers who have no completed order. Reference: 25 customers.
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: Common Table Expression (CTE)
-- One CTE computes each customer's total completed spend and a second computes the average of those totals,
-- the main query returns customers whose total is above the average.
-- Reference: 550 customers, average INR 2,552.06.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total_spend
    FROM customer_spend
)
SELECT cs.customer_id, ROUND(cs.total_spend, 2) AS total_spend
FROM customer_spend cs, avg_spend av
WHERE cs.total_spend > av.avg_total_spend;

-- Q6: Ranking window functions
-- For each customer with completed spend, compute total spend and then
-- RANK(), DENSE_RANK() and ROW_NUMBER() using OVER (PARTITION BY city ORDER BY total_spend DESC).
-- Reference: exactly 8 rows have RANK() different from ROW_NUMBER().
WITH cust_city_spend AS (
    SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM cust_city_spend;

-- Q7: Running total
-- Daily completed revenue (day is first 10 characters of order_ts) with cumulative revenue column
-- computed as SUM(revenue) OVER (ORDER BY day).
-- Reference: 183 daily rows, last cumulative value is INR 2,998,668.80.
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
