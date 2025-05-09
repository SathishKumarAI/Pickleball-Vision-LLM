#!/bin/bash

# Create directories
mkdir -p nginx/ssl

# Generate private key
openssl genrsa -out nginx/ssl/app.key 2048

# Generate CSR
openssl req -new -key nginx/ssl/app.key -out nginx/ssl/app.csr -subj "/CN=pickleball-vision.com"

# Generate self-signed certificate
openssl x509 -req -days 365 -in nginx/ssl/app.csr -signkey nginx/ssl/app.key -out nginx/ssl/app.crt

# Set proper permissions
chmod 600 nginx/ssl/app.key
chmod 644 nginx/ssl/app.crt

echo "SSL certificates generated successfully!"
echo "Key: nginx/ssl/app.key"
echo "Certificate: nginx/ssl/app.crt" 