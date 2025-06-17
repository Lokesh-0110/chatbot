# chatbot/views.py

from django.shortcuts import render
from django.http import JsonResponse
from .rag_logic import chatbot_pipeline
import json
from django.views.decorators.csrf import ensure_csrf_cookie # <--- IMPORT THIS

@ensure_csrf_cookie # <--- ADD THIS DECORATOR
def chat_view(request):
    if request.method == 'GET':
        # This decorator will ensure a CSRF cookie is sent with this response
        return render(request, 'chatbot/index.html')

    elif request.method == 'POST':
        # The POST request will now have the cookie to be validated
        try:
            data = json.loads(request.body)
            question = data.get('question')

            if not question:
                return JsonResponse({'error': 'No question provided.'}, status=400)

            # Change this back to .query()
            response = chatbot_pipeline.query(question)

            return JsonResponse(response)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON.'}, status=400)
        except Exception as e:
            # It's good practice to log the actual error on the server
            print(f"An error occurred: {e}") 
            return JsonResponse({'error': 'An internal server error occurred.'}, status=500)