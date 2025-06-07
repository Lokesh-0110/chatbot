# chatbot/views.py

from django.shortcuts import render
from django.http import JsonResponse
from .rag_logic import chatbot_pipeline
import json

def chat_view(request):
    if request.method == 'GET':
        # Just render the initial chat page
        return render(request, 'chatbot/index.html')

    elif request.method == 'POST':
        try:
            # Get the user's question from the POST request
            data = json.loads(request.body)
            question = data.get('question')

            if not question:
                return JsonResponse({'error': 'No question provided.'}, status=400)

            # Get the answer from the RAG pipeline
            response = chatbot_pipeline.query(question)

            return JsonResponse(response)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)