#!/usr/bin/env python3
"""
Debug script for OpenRouter API connection
Run this to test your API key and connection
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

def test_api_connection():
    """Test OpenRouter API connection"""
    print("🔬 OpenRouter API Connection Test")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Check API key
    print(f"1. API Key Status:")
    if not api_key:
        print("   ❌ OPENROUTER_API_KEY not found in environment")
        print("   💡 Make sure you have a .env file with OPENROUTER_API_KEY=your_key_here")
        return False
    
    print(f"   ✅ API key found")
    print(f"   📏 Length: {len(api_key)} characters")
    print(f"   🔍 Starts with correct format: {'✅' if api_key.startswith('sk-or-v1-') or api_key.startswith('sk-or-') else '❌'}")
    print(f"   🔒 Key preview: {api_key[:20]}...{api_key[-8:]}")
    
    if not (api_key.startswith('sk-or-v1-') or api_key.startswith('sk-or-')):
        print("   ❌ Invalid API key format")
        print("   💡 OpenRouter keys should start with 'sk-or-v1-' or 'sk-or-'")
        return False
    
    # Test connection
    print(f"\n2. Connection Test:")
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        print("   ✅ Client created successfully")
        
        # Test API call
        print("   🔄 Making test API call...")
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/debug-test",
                "X-Title": "Debug Test",
            },
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{
                "role": "user", 
                "content": "Hello! Please respond with 'API connection successful'"
            }],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"   ✅ API Response: {result}")
        print(f"   📊 Model used: {response.model}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Connection failed: {str(e)}")
        
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str:
            print("   💡 This is an authentication error. Your API key may be:")
            print("      - Invalid or expired")
            print("      - Not properly set in the .env file")
            print("      - Missing required permissions")
        elif "429" in error_str:
            print("   💡 Rate limit exceeded. Wait a few minutes and try again.")
        elif "400" in error_str:
            print("   💡 Bad request. The API parameters may be incorrect.")
        elif "503" in error_str:
            print("   💡 Service unavailable. The model may be temporarily down.")
        else:
            print("   💡 Unknown error. Check your internet connection and try again.")
            
        return False

def test_content_processing():
    """Test content processing with a sample text"""
    print(f"\n3. Content Processing Test:")
    
    sample_content = """
    This is a sample research paper abstract. 
    The study investigates the effects of machine learning algorithms on data processing efficiency.
    Our methodology involves comparing multiple algorithms across various datasets.
    Results show significant improvements in processing speed and accuracy.
    The findings have implications for future research in artificial intelligence.
    """
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("   ⏭️  Skipping content test - no API key")
        return False
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        print("   🔄 Testing summarization...")
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/debug-test",
                "X-Title": "Content Processing Test",
            },
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{
                "role": "user",
                "content": f"Please provide a brief summary of this research content:\n\n{sample_content}"
            }],
            max_tokens=200,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content
        print(f"   ✅ Summary generated:")
        print(f"   📄 {summary[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Content processing failed: {str(e)}")
        return False

def main():
    """Main debug function"""
    success = test_api_connection()
    
    if success:
        test_content_processing()
        print(f"\n🎉 All tests passed! Your setup is working correctly.")
        print(f"🚀 You can now run your Streamlit app with confidence.")
    else:
        print(f"\n❌ Tests failed. Please fix the issues above and try again.")
        print(f"\n🔧 Common solutions:")
        print(f"   1. Check your .env file exists and contains OPENROUTER_API_KEY")
        print(f"   2. Verify your API key at https://openrouter.ai/keys")
        print(f"   3. Make sure your API key has sufficient credits")
        print(f"   4. Try creating a new API key if the current one doesn't work")

if __name__ == "__main__":
    main()
