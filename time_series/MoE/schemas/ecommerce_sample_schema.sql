-- Example ecommerce analytics schema used by the SQL MoE orchestrator.
CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE,
    country_code CHAR(2),
    created_at TIMESTAMPTZ
);

CREATE TABLE products (
    product_id BIGINT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price NUMERIC(12, 2) NOT NULL
);

CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers(customer_id),
    order_status TEXT NOT NULL,
    order_timestamp TIMESTAMPTZ NOT NULL,
    currency CHAR(3) DEFAULT 'USD'
);

CREATE TABLE order_items (
    order_item_id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL REFERENCES orders(order_id),
    product_id BIGINT NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    sale_price NUMERIC(12, 2) NOT NULL
);

CREATE TABLE refunds (
    refund_id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL REFERENCES orders(order_id),
    refund_amount NUMERIC(12, 2) NOT NULL,
    refunded_at TIMESTAMPTZ NOT NULL
);

