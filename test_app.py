#!/usr/bin/env python3
"""
Simple test script for AskMeBot application.
Run this to verify the application is working correctly.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing imports...")
    
    try:
        from app.main import app
        print("✅ FastAPI app imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import main app: {e}")
        return False
    
    try:
        from app.groq_router import query_llm
        print("✅ Groq router imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import groq router: {e}")
        return False
    
    try:
        from app.chat_parser import parse_llm_output, parse_markdown_to_html
        print("✅ Chat parser imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import chat parser: {e}")
        return False
    
    return True

def test_markdown_parsing():
    """Test markdown parsing functionality."""
    print("\n🧪 Testing markdown parsing...")
    
    try:
        from app.chat_parser import parse_markdown_to_html
        
        test_cases = [
            ("**bold text**", "<strong>bold text</strong>"),
            ("*italic text*", "<em>italic text</em>"),
            ("`code text`", "<code class='bg-gray-800 text-yellow-400 px-1 py-0.5 rounded text-sm'>code text</code>"),
            ("# Heading", "<h1 class='text-xl font-bold text-yellow-400 mb-2'>Heading</h1>"),
            ("## Subheading", "<h2 class='text-lg font-bold text-yellow-300 mb-2'>Subheading</h2>"),
        ]
        
        for input_text, expected in test_cases:
            result = parse_markdown_to_html(input_text)
            # Normalize quotes for comparison
            expected_normalized = expected.replace("'", '"')
            if expected in result or expected_normalized in result:
                print(f"✅ '{input_text}' → parsed correctly")
            else:
                print(f"❌ '{input_text}' → expected '{expected}', got '{result}'")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Markdown parsing test failed: {e}")
        return False

def test_environment():
    """Test environment configuration."""
    print("\n🧪 Testing environment...")
    
    # Check if .env file exists
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found (create one with GROQ_API_KEY)")
    
    # Check for required environment variables
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key and groq_key != 'demo-key-for-github-pages':
        print("✅ GROQ_API_KEY is set")
    else:
        print("⚠️  GROQ_API_KEY not set or using demo key")
    
    return True

def test_fastapi_app():
    """Test FastAPI application structure."""
    print("\n🧪 Testing FastAPI app...")
    
    try:
        from app.main import app
        
        # Check if app is properly configured
        if hasattr(app, 'routes'):
            print(f"✅ FastAPI app has {len(app.routes)} routes")
        else:
            print("❌ FastAPI app missing routes")
            return False
        
        # Check for required endpoints
        route_paths = [route.path for route in app.routes]
        required_endpoints = ['/', '/health', '/chat', '/generate-title']
        
        for endpoint in required_endpoints:
            if endpoint in route_paths:
                print(f"✅ Endpoint {endpoint} found")
            else:
                print(f"❌ Endpoint {endpoint} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ FastAPI app test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🦇 AskMeBot Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_markdown_parsing,
        test_environment,
        test_fastapi_app,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AskMeBot is ready to go!")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
