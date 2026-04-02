from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import ChatSession, ChatMessage
from .rag import retrieve_context
from .llm import generate_answer



@login_required
def chat_list(request):
    chats = ChatSession.objects.filter(user=request.user).order_by("-updated_at")

    if chats.exists():
        return redirect("chat_detail", chat_id=chats.first().id)

    return render(request, "chat/chat_list.html", {"chats": chats})



@login_required
def create_chat(request):
    chat = ChatSession.objects.create(
        user=request.user,
        title="New Conversation"
    )
    return redirect("chat_detail", chat_id=chat.id)



@login_required
def chat_detail(request, chat_id):
    chat = get_object_or_404(ChatSession, id=chat_id, user=request.user)

    chats = ChatSession.objects.filter(user=request.user).order_by("-updated_at")
    messages = chat.messages.order_by("created_at")

    if request.method == "POST":
        user_input = request.POST.get("message", "").strip()

        if user_input:
            
            ChatMessage.objects.create(
                chat=chat,
                role="user",
                content=user_input
            )

            
            if messages.count() == 0:
                new_title = user_input[:35] + ("..." if len(user_input) > 35 else "")
                chat.title = new_title

            chat.save()

            try:
                user_lower = user_input.lower()

                
                if user_lower in ["hi", "hello", "hey"]:
                    ai_response = "Hello! How can I assist you with Indian law today?"

                else:
                    
                    result = retrieve_context(user_input)

                    
                    if isinstance(result, dict) and result.get("type") == "bare_act":
                        ai_response = f"📘 {result['content']}"

                    
                    elif isinstance(result, dict) and result.get("type") == "semantic":
                        prompt = f"""
Context:
{result['content']}

Question:
{user_input}

Answer completely but concisely. Include all punishments mentioned.
"""
                        ai_response = generate_answer(prompt)

                    
                    else:
                        ai_response = (
                            "I can assist with Indian legal queries. "
                            "Please ask a specific legal question."
                        )

            except Exception as e:
                ai_response = f"⚠ System Error: {str(e)}"

            
            ChatMessage.objects.create(
                chat=chat,
                role="assistant",
                content=ai_response
            )

        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            from django.http import JsonResponse
            return JsonResponse({"response": ai_response})

        return redirect("chat_detail", chat_id=chat.id)

    return render(request, "chat/chat_detail.html", {
        "chat": chat,
        "messages": messages,
        "chats": chats,
        "active_chat": chat
    })