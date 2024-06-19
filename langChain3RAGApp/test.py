import openai

def test_openai():
    try:
        openai_api_key = 'your_openai_api_key'  # Replace with your actual API key
        openai.api_key = openai_api_key
        response = openai.Completion.create(
            engine="davinci",
            prompt="Say this is a test",
            max_tokens=5
        )
        print(response)
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except Exception as e:
        print(f"Error: {e}")

test_openai()
