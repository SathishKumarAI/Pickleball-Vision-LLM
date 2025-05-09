#!/bin/bash

# Create required directories
mkdir -p deployment/monitoring/{prometheus,grafana,alertmanager,nginx}/{data,config}

# Generate SSL certificates
./scripts/generate-ssl.sh

# Create basic auth for Prometheus
htpasswd -bc deployment/monitoring/nginx/config/.htpasswd admin ${PROMETHEUS_PASSWORD}

# Set up Grafana admin password
export GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-"admin"}

# Start monitoring stack
docker-compose -f deployment/docker-compose.prod.yml up -d prometheus grafana alertmanager node-exporter

echo "Monitoring stack setup complete!"
echo "Grafana: https://localhost:443/grafana (admin/${GRAFANA_ADMIN_PASSWORD})"
echo "Prometheus: https://localhost:443/prometheus"
echo "AlertManager: https://localhost:443/alertmanager" 