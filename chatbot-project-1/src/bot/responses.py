def get_response(message):
    responses = {
        "hola": "¡Hola! ¿Cómo puedo ayudarte hoy?",
        "adiós": "¡Hasta luego! Que tengas un buen día.",
        "gracias": "¡De nada! Si necesitas algo más, no dudes en preguntar.",
        "ayuda": "Claro, ¿en qué necesitas ayuda?",
    }
    
    return responses.get(message.lower(), "Lo siento, no entiendo tu mensaje.")