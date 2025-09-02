#!/bin/bash

# AskMeBot Deployment Script
# This script helps deploy AskMeBot to various platforms

set -e  # Exit on any error

echo "🦇 AskMeBot Deployment Script"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found. Creating one..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    python test_app.py
    if [ $? -eq 0 ]; then
        print_success "All tests passed"
    else
        print_error "Tests failed"
        exit 1
    fi
}

# Check environment
check_env() {
    print_status "Checking environment..."
    
    if [ ! -f ".env" ]; then
        print_warning ".env file not found"
        print_status "Creating .env file from template..."
        cat > .env << EOF
# AskMeBot Environment Configuration
GROQ_API_KEY=your_groq_api_key_here
EOF
        print_warning "Please edit .env file and add your GROQ_API_KEY"
        print_status "You can get your API key from: https://console.groq.com/keys"
    else
        print_success ".env file found"
    fi
}

# Start the application
start_app() {
    print_status "Starting AskMeBot..."
    print_status "Server will be available at: http://localhost:8000"
    print_status "Press Ctrl+C to stop the server"
    echo ""
    
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}

# Main deployment function
deploy() {
    case "${1:-local}" in
        "local")
            print_status "Deploying locally..."
            check_venv
            install_deps
            check_env
            run_tests
            start_app
            ;;
        "test")
            print_status "Running tests only..."
            check_venv
            install_deps
            run_tests
            ;;
        "install")
            print_status "Installing dependencies only..."
            check_venv
            install_deps
            print_success "Installation complete"
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  local     Deploy locally (default)"
            echo "  test      Run tests only"
            echo "  install   Install dependencies only"
            echo "  help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                # Deploy locally"
            echo "  $0 local          # Deploy locally"
            echo "  $0 test           # Run tests"
            echo "  $0 install        # Install dependencies"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run the deployment
deploy "$@"
