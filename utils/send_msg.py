import requests
import urllib.parse
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

def send_whatsapp_message(message: str) -> None:
    """
    Sends a WhatsApp message using CallMeBot API.

    Args:
        message (str): The message to send.

    Raises:
        ValueError: If required environment variables are not set.
    """
    phone_number = os.getenv("PHONE_NUMBER", None)
    if not phone_number:
        raise ValueError("PHONE_NUMBER environment variable is not set.")

    api_key = os.getenv("CMB_API_KEY", None)
    if not api_key:
        raise ValueError("CMB_API_KEY environment variable is not set.")

    url = f"https://api.callmebot.com/whatsapp.php?phone={phone_number}&text={urllib.parse.quote(message)}&apikey={api_key}"

    response = requests.get(url)

    if response.status_code == 200:
        logger.info("Message sent successfully via WhatsApp.")
    else:
        logger.error(f"Error al enviar el mensaje: {response}")
