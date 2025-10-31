#!/bin/bash
# Deployment script for AI Bundle Creator Python Server
# This script helps deploy the application with CockroachDB

set -e  # Exit on error

echo "🚀 AI Bundle Creator - CockroachDB Deployment Script"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if .env file exists
if [ ! -f .env ]; then
    print_error ".env file not found!"
    echo "Please create a .env file with your CockroachDB connection string."
    echo "See .env.example for reference."
    exit 1
fi

print_success ".env file found"

# Check if DATABASE_URL is set
if ! grep -q "DATABASE_URL=" .env; then
    print_error "DATABASE_URL not found in .env file!"
    exit 1
fi

print_success "DATABASE_URL configured"

# Test database connection
echo ""
echo "📊 Testing database connection..."
if source venv/bin/activate && python test_cockroach_connection.py 2>&1 | grep -q "successfully"; then
    print_success "Database connection test passed"
else
    print_error "Database connection test failed"
    echo "Please check your DATABASE_URL and CockroachDB cluster status."
    exit 1
fi

echo ""
echo "🎯 Deployment Options:"
echo "1. Docker (recommended for production)"
echo "2. Local with Gunicorn"
echo "3. Development server (uvicorn)"
echo ""
read -p "Select deployment option (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🐳 Building Docker image..."
        docker build -t ai-bundle-creator:latest .
        print_success "Docker image built"

        echo ""
        echo "Starting container..."
        docker run -d \
            --name ai-bundle-creator \
            -p 8080:8080 \
            --env-file .env \
            ai-bundle-creator:latest

        print_success "Container started on port 8080"
        echo ""
        echo "View logs: docker logs -f ai-bundle-creator"
        echo "Stop container: docker stop ai-bundle-creator"
        echo "Remove container: docker rm ai-bundle-creator"
        ;;

    2)
        echo ""
        echo "🚀 Starting production server with Gunicorn..."
        if [ ! -d "venv" ]; then
            print_warning "Virtual environment not found. Creating..."
            python3 -m venv venv
            source venv/bin/activate
            pip install -r requirements.txt
        else
            source venv/bin/activate
        fi

        print_success "Starting server on port 8080..."
        gunicorn main:app \
            -w 4 \
            -k uvicorn.workers.UvicornWorker \
            --bind 0.0.0.0:8080 \
            --timeout 120 \
            --access-logfile - \
            --error-logfile -
        ;;

    3)
        echo ""
        echo "🔧 Starting development server..."
        if [ ! -d "venv" ]; then
            print_warning "Virtual environment not found. Creating..."
            python3 -m venv venv
            source venv/bin/activate
            pip install -r requirements.txt
        else
            source venv/bin/activate
        fi

        print_success "Starting development server on port 5000..."
        uvicorn main:app --reload --host 0.0.0.0 --port 5000
        ;;

    *)
        print_error "Invalid option selected"
        exit 1
        ;;
esac
