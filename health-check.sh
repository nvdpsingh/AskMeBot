#!/bin/bash

# AskMeBot Health Check Script
# This script checks if the application is running correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if server is running
check_server() {
    local url="${1:-http://localhost:8000}"
    print_status "Checking server at $url..."
    
    if curl -s -f "$url/health" > /dev/null 2>&1; then
        print_success "Server is running and healthy"
        return 0
    else
        print_error "Server is not responding"
        return 1
    fi
}

# Check GitHub Pages deployment
check_github_pages() {
    local url="https://nvdpsingh.github.io/AskMeBot/"
    print_status "Checking GitHub Pages deployment at $url..."
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        print_success "GitHub Pages is accessible"
        return 0
    else
        print_warning "GitHub Pages might not be deployed yet or is not accessible"
        return 1
    fi
}

# Check API endpoints
check_api() {
    local base_url="${1:-http://localhost:8000}"
    print_status "Checking API endpoints..."
    
    # Check health endpoint
    if curl -s -f "$base_url/health" | grep -q "healthy"; then
        print_success "Health endpoint working"
    else
        print_error "Health endpoint failed"
        return 1
    fi
    
    # Check main page
    if curl -s -f "$base_url/" > /dev/null 2>&1; then
        print_success "Main page accessible"
    else
        print_error "Main page not accessible"
        return 1
    fi
    
    return 0
}

# Main health check function
main() {
    echo "🦇 AskMeBot Health Check"
    echo "========================"
    
    local all_good=true
    
    # Check local server if running
    if pgrep -f "uvicorn.*app.main:app" > /dev/null; then
        print_status "Local server detected, checking..."
        if ! check_server "http://localhost:8000"; then
            all_good=false
        fi
        
        if ! check_api "http://localhost:8000"; then
            all_good=false
        fi
    else
        print_warning "Local server not running"
    fi
    
    # Check GitHub Pages
    if ! check_github_pages; then
        all_good=false
    fi
    
    echo ""
    echo "========================"
    
    if [ "$all_good" = true ]; then
        print_success "All health checks passed! 🎉"
        echo ""
        echo "🌐 Local: http://localhost:8000"
        echo "🌐 GitHub Pages: https://nvdpsingh.github.io/AskMeBot/"
        return 0
    else
        print_error "Some health checks failed"
        echo ""
        echo "💡 Troubleshooting tips:"
        echo "   - Start local server: ./deploy.sh local"
        echo "   - Check GitHub Actions: https://github.com/nvdpsingh/AskMeBot/actions"
        echo "   - Verify Pages settings: https://github.com/nvdpsingh/AskMeBot/settings/pages"
        return 1
    fi
}

# Run the health check
main "$@"
