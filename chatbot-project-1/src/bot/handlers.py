import re
import random

class MessageHandler:
    def __init__(self):
        self.responses = [
            {"response": "Hola. ¿Qué tal la vida?", "keywords": ['hola', 'saludos', 'buenas', 'que tal', 'oe causa', 'manito'], "single_response": True},
            {"response": "Estoy bien. ¿Y tú?", "keywords": ['como', 'estas', 'buenas', 'que tal', 'todo bien', 'como vas','bien y tu'], "required_words": ['como']},
            {"response": "Que bueno, ¿Qué deseas hacer hoy? Recuerda que poniendo el comando /help podrás ver todas mis opciones", "keywords": ['necesito', 'me siento bien', 'en fin'], "required_words": ['bien']},
            {"response": "Lista de comandos: \n", "keywords": ['help'], "required_words": ['help']},
            {"response": "Estamos en Senati Chiclayo", "keywords": ['ubicados', 'direccion', 'donde', 'ubicacion', 'por donde queda', 'donde estan'], "single_response": True}
        ]
        self.positive_responses = [
            "¡Genial! Me alegra escuchar eso.",
            "¡Qué bien! ¿En qué más puedo ayudarte?",
            "¡Excelente! ¿Algo más que necesites?"
        ]
        self.negative_responses = [
            "Lo siento, no entiendo tu pregunta.",
            "No estoy seguro de cómo responder a eso.",
            "Podrías intentar preguntar de otra manera."
        ]

    def handle_message(self, message):
        cleaned_message = self._clean_message(message)
        for response in self.responses:
            if self._message_matches(cleaned_message, response):
                return response["response"]
        return self.response_negative()

    def _clean_message(self, message):
        # Eliminar símbolos y convertir a minúsculas
        return re.sub(r'[^\w\s]', '', message).lower()

    def _message_matches(self, message, response):
        message_words = message.split()
        if response.get("single_response"):
            return any(word in message_words for word in response["keywords"])
        if response.get("required_words"):
            return all(word in message_words for word in response["required_words"]) and any(word in message_words for word in response["keywords"])
        return False

    def response_negative(self):
        return random.choice(self.negative_responses)

    def response_positive(self):
        return random.choice(self.positive_responses)